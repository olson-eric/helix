use anyhow::{bail, Context, Result};
use arc_swap::ArcSwap;
use gix::filter::plumbing::driver::apply::Delay;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use gix::bstr::ByteSlice;
use gix::diff::Rewrites;
use gix::dir::entry::Status;
use gix::objs::tree::EntryKind;
use gix::sec::trust::DefaultForLevel;
use gix::status::{
    index_worktree::Item,
    plumbing::index_as_worktree::{Change, EntryStatus},
    UntrackedFiles,
};
use gix::{Commit, ObjectId, Repository, ThreadSafeRepository};

use crate::{FileChange, StatusScope};

#[cfg(test)]
mod test;

#[inline]
fn get_repo_dir(file: &Path) -> Result<&Path> {
    file.parent().context("file has no parent directory")
}

pub fn get_diff_base(file: &Path, trust_full: bool) -> Result<Vec<u8>> {
    get_diff_base_impl(file, None, trust_full)
}

/// Like [`get_diff_base`], but reads the file as of the merge base of `rev`
/// and `HEAD` (`git diff <rev>...` semantics) instead of `HEAD` itself.
pub fn get_diff_base_at(file: &Path, rev: &str, trust_full: bool) -> Result<Vec<u8>> {
    get_diff_base_impl(file, Some(rev), trust_full)
}

/// The commit a diff is computed against: `HEAD`, or for a given revision
/// the merge base of that revision and `HEAD`, mirroring the "three dot"
/// (`git diff <rev>...`) semantics used by pull-request views.
fn base_commit<'a>(repo: &'a Repository, rev: Option<&str>) -> Result<Commit<'a>> {
    let head = repo.head_commit()?;
    let Some(rev) = rev else {
        return Ok(head);
    };
    let commit = repo
        .rev_parse_single(rev)
        .with_context(|| format!("cannot resolve revision '{rev}'"))?
        .object()?
        .peel_to_kind(gix::object::Kind::Commit)
        .with_context(|| format!("revision '{rev}' does not point to a commit"))?
        .into_commit();
    let base_id = repo
        .merge_base(commit.id, head.id)
        .with_context(|| format!("no merge base between '{rev}' and HEAD"))?;
    Ok(base_id.object()?.try_into_commit()?)
}

fn get_diff_base_impl(file: &Path, rev: Option<&str>, trust_full: bool) -> Result<Vec<u8>> {
    debug_assert!(!file.exists() || file.is_file());
    debug_assert!(file.is_absolute());
    let file = gix::path::realpath(file).context("resolve symlinks")?;

    // TODO cache repository lookup

    let repo_dir = get_repo_dir(&file)?;
    let repo = open_repo(repo_dir, trust_full)
        .context("failed to open git repo")?
        .to_thread_local();
    let base = base_commit(&repo, rev)?;
    let file_oid = find_file_in_commit(&repo, &base, &file)?;

    let file_object = repo.find_object(file_oid)?;
    let data = file_object.detach().data;
    // Get the actual data that git would make out of the git object.
    // This will apply the user's git config or attributes like crlf conversions.
    //
    // The whole filter pipeline still runs in untrusted (`Trust::Reduced`) mode so built-in
    // conversions like autocrlf keep working, but gix drops `filter.*.clean` / `filter.*.smudge`
    // drivers defined in untrusted (repository-local) config, so those external programs are not
    // executed unless the workspace was explicitly trusted. This relies on `open_repo` forcing the
    // trust level instead of letting gix re-derive it from `.git` ownership; see the note there.
    if let Some(work_dir) = repo.workdir() {
        let rela_path = file.strip_prefix(work_dir)?;
        let rela_path = gix::path::try_into_bstr(rela_path)?;
        let (mut pipeline, _) = repo.filter_pipeline(None)?;
        let mut worktree_outcome =
            pipeline.convert_to_worktree(&data, rela_path.as_ref(), Delay::Forbid)?;
        let mut buf = Vec::with_capacity(data.len());
        worktree_outcome.read_to_end(&mut buf)?;
        Ok(buf)
    } else {
        Ok(data)
    }
}

pub fn get_current_head_name(file: &Path, trust_full: bool) -> Result<Arc<ArcSwap<Box<str>>>> {
    debug_assert!(!file.exists() || file.is_file());
    debug_assert!(file.is_absolute());
    let file = gix::path::realpath(file).context("resolve symlinks")?;

    let repo_dir = get_repo_dir(&file)?;
    let repo = open_repo(repo_dir, trust_full)
        .context("failed to open git repo")?
        .to_thread_local();
    let head_ref = repo.head_ref()?;
    let head_commit = repo.head_commit()?;

    let name = match head_ref {
        Some(reference) => reference.name().shorten().to_string(),
        None => head_commit.id.to_hex_with_len(8).to_string(),
    };

    Ok(Arc::new(ArcSwap::from_pointee(name.into_boxed_str())))
}

pub fn for_each_changed_file(
    cwd: &Path,
    trust_full: bool,
    scope: &StatusScope,
    f: impl Fn(Result<FileChange>) -> bool,
) -> Result<()> {
    status(&open_repo(cwd, trust_full)?.to_thread_local(), scope, f)
}

fn open_repo(path: &Path, trust_full: bool) -> Result<ThreadSafeRepository> {
    // `trust_full` is the workspace-trust decision made by the caller, and it must be the
    // authority on the gix trust level. gix's own discovery (`discover_*`) ignores a
    // caller-supplied trust level: it always re-derives trust from `.git` ownership, so a malicious
    // `.git/config` in a user-owned directory would be opened as `Trust::Full` regardless of our
    // gate. Worse, the GIT_DIR-environment branch of that discovery panics because it never sets a
    // trust level at all. So we split discovery from opening: find the repository path ourselves,
    // then `open_opts(..).with(trust)`, which forces the trust level and skips gix's ownership
    // check. Under `Trust::Reduced`, gix then refuses to honor untrusted repository-local config
    // such as `filter.*` smudge/clean drivers.

    let trust = if trust_full {
        gix::sec::Trust::Full
    } else {
        gix::sec::Trust::Reduced
    };

    // On Windows various configuration options are bundled as part of the git installation. The
    // lookup is expensive; only do it there.
    let config = gix::open::permissions::Config {
        system: true,
        git: true,
        user: true,
        env: true,
        includes: true,
        git_binary: cfg!(windows),
    };

    let permissions = gix::open::Permissions {
        config,
        ..gix::open::Permissions::default_for_level(trust)
    };

    let discover_options = gix::discover::upwards::Options {
        dot_git_only: true,
        ..Default::default()
    };
    let (repo_path, _trust_from_ownership) = gix::discover::upwards_opts(path, discover_options)
        .context("failed to discover git repo")?;
    let (git_dir, _work_dir) = repo_path.into_repository_and_work_tree_directories();

    let options = gix::open::Options::default()
        .permissions(permissions)
        // `git_dir` is the discovered `.git` directory (or a linked-worktree git dir), so open it
        // as-is rather than letting gix append `.git` again.
        .open_path_as_is(true)
        .with(trust);

    Ok(ThreadSafeRepository::open_opts(git_dir, options)?)
}

/// Emulates the result of running `git status` from the command line.
fn status(
    repo: &Repository,
    scope: &StatusScope,
    f: impl Fn(Result<FileChange>) -> bool,
) -> Result<()> {
    let work_dir = repo
        .workdir()
        .ok_or_else(|| anyhow::anyhow!("working tree not found"))?
        .to_path_buf();

    let rewrites = Rewrites {
        copies: None,
        percentage: Some(0.5),
        limit: 1000,
        ..Default::default()
    };
    let status_platform = repo
        .status(gix::progress::Discard)?
        // Here we discard the `status.showUntrackedFiles` config, as it makes little sense in
        // our case to not list new (untracked) files. We could have respected this config
        // if the default value weren't `Collapsed` though, as this default value would render
        // the feature unusable to many.
        .untracked_files(UntrackedFiles::Files)
        // Turn on file rename detection, which is off by default.
        .index_worktree_rewrites(Some(rewrites));

    // No filtering based on path
    let empty_patterns = vec![];

    if let StatusScope::HeadToWorktree | StatusScope::MergeBaseToWorktree(_) = scope {
        // Also compare a base tree against the index, so that staged (and,
        // for a revision base, committed) changes are reported as well
        // rather than only index-vs-worktree ones.
        let mut status_platform = status_platform
            .tree_index_track_renames(gix::status::tree_index::TrackRenames::Given(rewrites));
        if let StatusScope::MergeBaseToWorktree(rev) = scope {
            let base = match base_commit(repo, Some(rev)) {
                Ok(base) => base,
                Err(err) => {
                    // Report the resolution failure to the callback rather
                    // than failing the provider, so the message reaches the
                    // user instead of a generic "no provider" error.
                    f(Err(err));
                    return Ok(());
                }
            };
            status_platform = status_platform.head_tree(base.tree_id()?.detach());
        }
        let status_iter = status_platform.into_iter(empty_patterns)?;
        for item in status_iter {
            let Ok(item) = item.map_err(|err| f(Err(err.into()))) else {
                continue;
            };
            let change = match item {
                gix::status::Item::IndexWorktree(item) => index_worktree_change(&work_dir, item)?,
                gix::status::Item::TreeIndex(change) => tree_index_change(&work_dir, change)?,
            };
            let Some(change) = change else {
                continue;
            };
            if !f(Ok(change)) {
                break;
            }
        }
    } else {
        let status_iter = status_platform.into_index_worktree_iter(empty_patterns)?;
        for item in status_iter {
            let Ok(item) = item.map_err(|err| f(Err(err.into()))) else {
                continue;
            };
            let Some(change) = index_worktree_change(&work_dir, item)? else {
                continue;
            };
            if !f(Ok(change)) {
                break;
            }
        }
    }

    Ok(())
}

fn index_worktree_change(work_dir: &Path, item: Item) -> Result<Option<FileChange>> {
    let change = match item {
        Item::Modification {
            rela_path, status, ..
        } => {
            let path = work_dir.join(rela_path.to_path()?);
            match status {
                EntryStatus::Conflict { .. } => FileChange::Conflict { path },
                EntryStatus::Change(Change::Removed) => FileChange::Deleted { path },
                EntryStatus::Change(Change::Modification { .. }) => FileChange::Modified { path },
                // Files marked with `git add --intent-to-add`. Such files
                // still show up as new in `git status`, so it's appropriate
                // to show them the same way as untracked files in the
                // "changed file" picker. One example of this being used
                // is Jujutsu, a Git-compatible VCS. It marks all new files
                // with `--intent-to-add` automatically.
                EntryStatus::IntentToAdd => FileChange::Untracked { path },
                _ => return Ok(None),
            }
        }
        Item::DirectoryContents { entry, .. } if entry.status == Status::Untracked => {
            FileChange::Untracked {
                path: work_dir.join(entry.rela_path.to_path()?),
            }
        }
        Item::Rewrite {
            source,
            dirwalk_entry,
            ..
        } => FileChange::Renamed {
            from_path: work_dir.join(source.rela_path().to_path()?),
            to_path: work_dir.join(dirwalk_entry.rela_path.to_path()?),
        },
        _ => return Ok(None),
    };
    Ok(Some(change))
}

/// Maps a `HEAD`-vs-index (staged) change.
fn tree_index_change(
    work_dir: &Path,
    change: gix::diff::index::Change,
) -> Result<Option<FileChange>> {
    use gix::diff::index::ChangeRef;
    let change = match change {
        ChangeRef::Addition { location, .. } => FileChange::Untracked {
            path: work_dir.join(location.to_path()?),
        },
        ChangeRef::Deletion { location, .. } => FileChange::Deleted {
            path: work_dir.join(location.to_path()?),
        },
        ChangeRef::Modification { location, .. } => FileChange::Modified {
            path: work_dir.join(location.to_path()?),
        },
        ChangeRef::Rewrite {
            source_location,
            location,
            ..
        } => FileChange::Renamed {
            from_path: work_dir.join(source_location.to_path()?),
            to_path: work_dir.join(location.to_path()?),
        },
    };
    Ok(Some(change))
}

/// Finds the object that contains the contents of a file at a specific commit.
fn find_file_in_commit(repo: &Repository, commit: &Commit, file: &Path) -> Result<ObjectId> {
    let repo_dir = repo.workdir().context("repo has no worktree")?;
    let rel_path = file.strip_prefix(repo_dir)?;
    let tree = commit.tree()?;
    let tree_entry = tree
        .lookup_entry_by_path(rel_path)?
        .context("file is untracked")?;
    match tree_entry.mode().kind() {
        // not a file, everything is new, do not show diff
        mode @ (EntryKind::Tree | EntryKind::Commit | EntryKind::Link) => {
            bail!("entry at {} is not a file but a {mode:?}", file.display())
        }
        // found a file
        EntryKind::Blob | EntryKind::BlobExecutable => Ok(tree_entry.object_id()),
    }
}
