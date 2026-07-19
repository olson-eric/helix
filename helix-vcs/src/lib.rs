//! `helix_vcs` provides types for working with diffs from a Version Control System (VCS).
//! Currently `git` is the only supported provider for diffs, but this architecture allows
//! for other providers to be added in the future.

use anyhow::{anyhow, bail, Result};
use arc_swap::ArcSwap;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(feature = "git")]
mod git;

mod diff;

pub use diff::{DiffHandle, Hunk};

mod status;

pub use status::FileChange;

/// Which changes to report when iterating over changed files.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum StatusScope {
    /// Changes between the index and the worktree only.
    #[default]
    IndexToWorktree,
    /// All changes between `HEAD` and the worktree, including staged
    /// changes. A file changed both in the index and the worktree may be
    /// reported twice.
    HeadToWorktree,
    /// All changes between the merge base of the given revision and `HEAD`,
    /// and the worktree — "three dot" (`git diff <rev>...`) semantics, like
    /// a pull-request view.
    MergeBaseToWorktree(String),
}

/// Contains all active diff providers. Diff providers are compiled in via features. Currently
/// only `git` is supported.
#[derive(Clone)]
pub struct DiffProviderRegistry {
    providers: Vec<DiffProvider>,
}

impl DiffProviderRegistry {
    /// Get the given file from the VCS. This provides the unedited document as a "base"
    /// for a diff to be created.
    pub fn get_diff_base(&self, file: &Path, trust_full: bool) -> Option<Vec<u8>> {
        self.providers
            .iter()
            .find_map(|provider| match provider.get_diff_base(file, trust_full) {
                Ok(res) => Some(res),
                Err(err) => {
                    log::debug!("{err:#?}");
                    log::debug!("failed to open diff base for {}", file.display());
                    None
                }
            })
    }

    /// Like [`Self::get_diff_base`], but reads the file as of the merge base
    /// of `rev` and `HEAD` instead of `HEAD` itself.
    pub fn get_diff_base_at(&self, file: &Path, rev: &str, trust_full: bool) -> Option<Vec<u8>> {
        self.providers.iter().find_map(|provider| {
            match provider.get_diff_base_at(file, rev, trust_full) {
                Ok(res) => Some(res),
                Err(err) => {
                    log::debug!("{err:#?}");
                    log::debug!("failed to open diff base at {} for {}", rev, file.display());
                    None
                }
            }
        })
    }

    /// Get the current name of the current [HEAD](https://stackoverflow.com/questions/2304087/what-is-head-in-git).
    pub fn get_current_head_name(
        &self,
        file: &Path,
        trust_full: bool,
    ) -> Option<Arc<ArcSwap<Box<str>>>> {
        self.providers.iter().find_map(|provider| {
            match provider.get_current_head_name(file, trust_full) {
                Ok(res) => Some(res),
                Err(err) => {
                    log::debug!("{err:#?}");
                    log::debug!("failed to obtain current head name for {}", file.display());
                    None
                }
            }
        })
    }

    /// Fire-and-forget changed file iteration. Runs everything in a background task. Keeps
    /// iteration until `on_change` returns `false`.
    pub fn for_each_changed_file(
        self,
        cwd: PathBuf,
        trust_full: bool,
        scope: StatusScope,
        f: impl Fn(Result<FileChange>) -> bool + Send + 'static,
    ) {
        tokio::task::spawn_blocking(move || {
            if self
                .providers
                .iter()
                .find_map(|provider| {
                    provider
                        .for_each_changed_file(&cwd, trust_full, &scope, &f)
                        .ok()
                })
                .is_none()
            {
                f(Err(anyhow!("no diff provider returns success")));
            }
        });
    }
}

impl Default for DiffProviderRegistry {
    fn default() -> Self {
        // currently only git is supported
        // TODO make this configurable when more providers are added
        let providers = vec![
            #[cfg(feature = "git")]
            DiffProvider::Git,
            DiffProvider::None,
        ];
        DiffProviderRegistry { providers }
    }
}

/// A union type that includes all types that implement [DiffProvider]. We need this type to allow
/// cloning [DiffProviderRegistry] as `Clone` cannot be used in trait objects.
///
/// `Copy` is simply to ensure the `clone()` call is the simplest it can be.
#[derive(Copy, Clone)]
enum DiffProvider {
    #[cfg(feature = "git")]
    Git,
    None,
}

impl DiffProvider {
    fn get_diff_base(&self, file: &Path, trust_full: bool) -> Result<Vec<u8>> {
        match self {
            #[cfg(feature = "git")]
            Self::Git => git::get_diff_base(file, trust_full),
            Self::None => bail!("No diff support compiled in"),
        }
    }

    fn get_diff_base_at(&self, file: &Path, rev: &str, trust_full: bool) -> Result<Vec<u8>> {
        match self {
            #[cfg(feature = "git")]
            Self::Git => git::get_diff_base_at(file, rev, trust_full),
            Self::None => bail!("No diff support compiled in"),
        }
    }

    fn get_current_head_name(
        &self,
        file: &Path,
        trust_full: bool,
    ) -> Result<Arc<ArcSwap<Box<str>>>> {
        match self {
            #[cfg(feature = "git")]
            Self::Git => git::get_current_head_name(file, trust_full),
            Self::None => bail!("No diff support compiled in"),
        }
    }

    fn for_each_changed_file(
        &self,
        cwd: &Path,
        trust_full: bool,
        scope: &StatusScope,
        f: impl Fn(Result<FileChange>) -> bool,
    ) -> Result<()> {
        match self {
            #[cfg(feature = "git")]
            Self::Git => git::for_each_changed_file(cwd, trust_full, scope, f),
            Self::None => bail!("No diff support compiled in"),
        }
    }
}
