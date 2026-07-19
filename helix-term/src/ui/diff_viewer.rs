//! A GitHub-style diff viewer for the whole workspace.
//!
//! `:diff` (or `<space>v`) opens a full-screen view of every change in the
//! repository (working tree vs. `HEAD`), rendered like a pull-request diff:
//! per-file sections with change stats, unified or side-by-side hunks,
//! intra-line change highlights and expandable context. Any diff line can be
//! opened in the editor at that exact position, optionally jumping straight
//! to the definition/references/type-definition of the symbol under the
//! cursor via LSP.

use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::Result;
use imara_diff::{Algorithm, IndentHeuristic, IndentLevel, InternedInput, Interner};

use helix_core::unicode::width::UnicodeWidthChar;
use helix_core::Selection;
use helix_vcs::{DiffProviderRegistry, FileChange, StatusScope};
use helix_view::{
    align_view,
    editor::Action,
    graphics::{Modifier, Rect, Style},
    input::{Event, MouseButton, MouseEvent, MouseEventKind},
    theme::Theme,
    Align, Editor,
};
use tui::buffer::Buffer as Surface;
use tui::widgets::{Block, Widget};

use crate::compositor::{Component, Compositor, Context, EventResult};
use crate::job::{self, Jobs};
use crate::{ctrl, key, shift};

/// Context lines shown around every change, like `git diff` / GitHub.
const CONTEXT_LINES: usize = 3;
/// Lines revealed by a single "expand context" step (GitHub uses 20).
const EXPAND_LINES: usize = 20;
/// Files with more changed lines than this start out collapsed,
/// mirroring GitHub's "large diffs are not rendered by default".
const AUTO_COLLAPSE_CHANGES: usize = 2000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileStatus {
    Added,
    Modified,
    Deleted,
    Renamed,
    Conflict,
}

impl FileStatus {
    fn letter(self) -> &'static str {
        match self {
            FileStatus::Added => "A",
            FileStatus::Modified => "M",
            FileStatus::Deleted => "D",
            FileStatus::Renamed => "R",
            FileStatus::Conflict => "C",
        }
    }

    fn theme_scope(self) -> &'static str {
        match self {
            FileStatus::Added => "diff.plus",
            FileStatus::Modified => "diff.delta",
            FileStatus::Deleted => "diff.minus",
            FileStatus::Renamed => "diff.delta.moved",
            FileStatus::Conflict => "diff.delta.conflict",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpKind {
    Context,
    Removed,
    Added,
}

const NO_LINE: usize = usize::MAX;

/// One aligned line of the diff: a context line (present on both sides),
/// a removed line (old side only) or an added line (new side only).
#[derive(Debug, Clone)]
struct Op {
    kind: OpKind,
    /// Line index into `FileDiff::old_lines`, `NO_LINE` for added lines.
    old: usize,
    /// Line index into `FileDiff::new_lines`, `NO_LINE` for removed lines.
    new: usize,
    /// Byte range within the line that actually changed (intra-line diff).
    hl: Option<Range<usize>>,
}

impl Op {
    fn context(old: usize, new: usize) -> Self {
        Op {
            kind: OpKind::Context,
            old,
            new,
            hl: None,
        }
    }

    fn removed(old: usize) -> Self {
        Op {
            kind: OpKind::Removed,
            old,
            new: NO_LINE,
            hl: None,
        }
    }

    fn added(new: usize) -> Self {
        Op {
            kind: OpKind::Added,
            old: NO_LINE,
            new,
            hl: None,
        }
    }
}

pub struct FileDiff {
    abs_path: PathBuf,
    display_path: String,
    old_display_path: Option<String>,
    status: FileStatus,
    binary: bool,
    old_lines: Vec<String>,
    new_lines: Vec<String>,
    /// Full alignment of the two files, in order.
    ops: Vec<Op>,
    /// Displayed hunks as disjoint, sorted ranges into `ops`
    /// (changes plus surrounding context; grows when context is expanded).
    hunks: Vec<Range<usize>>,
    additions: usize,
    deletions: usize,
    collapsed: bool,
}

impl FileDiff {
    /// `(first old line, old line count, first new line, new line count)`
    /// of a hunk, 1-based, as shown in `@@ -a,b +c,d @@` headers.
    fn hunk_line_info(&self, hunk: &Range<usize>) -> (usize, usize, usize, usize) {
        let mut old_start = 0;
        let mut old_count = 0;
        let mut new_start = 0;
        let mut new_count = 0;
        for op in &self.ops[hunk.clone()] {
            if op.old != NO_LINE {
                if old_count == 0 {
                    old_start = op.old + 1;
                }
                old_count += 1;
            }
            if op.new != NO_LINE {
                if new_count == 0 {
                    new_start = op.new + 1;
                }
                new_count += 1;
            }
        }
        (old_start, old_count, new_start, new_count)
    }

    /// Number of hidden (collapsed context) lines above the given hunk.
    fn gap_above(&self, hunk_idx: usize) -> usize {
        let prev_end = if hunk_idx == 0 {
            0
        } else {
            self.hunks[hunk_idx - 1].end
        };
        self.hunks[hunk_idx].start - prev_end
    }

    /// Number of hidden lines below the last hunk.
    fn gap_below(&self) -> usize {
        self.hunks
            .last()
            .map_or(0, |hunk| self.ops.len() - hunk.end)
    }
}

/// Split raw file contents into display lines (line terminators stripped).
fn split_lines(text: &str) -> Vec<String> {
    text.split_inclusive('\n')
        .map(|line| {
            line.strip_suffix('\n')
                .map(|line| line.strip_suffix('\r').unwrap_or(line))
                .unwrap_or(line)
                .to_string()
        })
        .collect()
}

fn normalize_whitespace(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    for word in line.split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word);
    }
    out
}

/// The byte ranges of `old` and `new` that differ, found by trimming the
/// common prefix and suffix (clamped to char boundaries). Returns `None`
/// when the lines have nothing in common (highlighting everything is noise).
fn intraline_ranges(old: &str, new: &str) -> Option<(Range<usize>, Range<usize>)> {
    if old == new {
        return None;
    }
    let mut prefix = old
        .bytes()
        .zip(new.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    while prefix > 0 && !(old.is_char_boundary(prefix) && new.is_char_boundary(prefix)) {
        prefix -= 1;
    }
    let max_suffix = old.len().min(new.len()) - prefix;
    let mut suffix = old
        .bytes()
        .rev()
        .zip(new.bytes().rev())
        .take_while(|(a, b)| a == b)
        .count()
        .min(max_suffix);
    while suffix > 0
        && !(old.is_char_boundary(old.len() - suffix) && new.is_char_boundary(new.len() - suffix))
    {
        suffix -= 1;
    }
    if prefix == 0 && suffix == 0 {
        return None;
    }
    Some((prefix..old.len() - suffix, prefix..new.len() - suffix))
}

/// Diff two files line-wise and produce the aligned op list, the displayed
/// hunk ranges (changes + `CONTEXT_LINES` of context, merged when they
/// overlap) and the addition/deletion counts.
fn compute_ops(
    old_lines: &[String],
    new_lines: &[String],
    ignore_whitespace: bool,
) -> (Vec<Op>, Vec<Range<usize>>, usize, usize) {
    // Diff normalized copies when ignoring whitespace; the token streams
    // stay line-aligned with the original contents either way.
    let normalized: Option<(Vec<String>, Vec<String>)> = ignore_whitespace.then(|| {
        (
            old_lines.iter().map(|l| normalize_whitespace(l)).collect(),
            new_lines.iter().map(|l| normalize_whitespace(l)).collect(),
        )
    });
    let (before_src, after_src) = match &normalized {
        Some((old, new)) => (old.as_slice(), new.as_slice()),
        None => (old_lines, new_lines),
    };

    let mut input: InternedInput<&str> = InternedInput {
        before: Vec::with_capacity(before_src.len()),
        after: Vec::with_capacity(after_src.len()),
        interner: Interner::new(before_src.len() + after_src.len()),
    };
    input.update_before(before_src.iter().map(String::as_str));
    input.update_after(after_src.iter().map(String::as_str));

    let mut diff = imara_diff::Diff::compute(Algorithm::Histogram, &input);
    diff.postprocess_with(
        &input.before,
        &input.after,
        IndentHeuristic::new(|token| IndentLevel::for_ascii_line(input.interner[token].bytes(), 4)),
    );

    let mut ops = Vec::with_capacity(old_lines.len().max(new_lines.len()));
    let mut change_ranges: Vec<Range<usize>> = Vec::new();
    let mut additions = 0;
    let mut deletions = 0;
    let mut old_pos = 0;
    let mut new_pos = 0;
    for hunk in diff.hunks() {
        while old_pos < hunk.before.start as usize {
            ops.push(Op::context(old_pos, new_pos));
            old_pos += 1;
            new_pos += 1;
        }
        let change_start = ops.len();
        let removed = hunk.before.len();
        let added = hunk.after.len();
        for old in hunk.before.clone() {
            ops.push(Op::removed(old as usize));
        }
        for new in hunk.after.clone() {
            ops.push(Op::added(new as usize));
        }
        deletions += removed;
        additions += added;
        old_pos = hunk.before.end as usize;
        new_pos = hunk.after.end as usize;

        // Pair the i-th removed line with the i-th added line for
        // intra-line highlights, like GitHub's word diff.
        for i in 0..removed.min(added) {
            let old_line = &old_lines[ops[change_start + i].old];
            let new_line = &new_lines[ops[change_start + removed + i].new];
            if let Some((old_hl, new_hl)) = intraline_ranges(old_line, new_line) {
                ops[change_start + i].hl = Some(old_hl);
                ops[change_start + removed + i].hl = Some(new_hl);
            }
        }
        change_ranges.push(change_start..ops.len());
    }
    while old_pos < old_lines.len() {
        ops.push(Op::context(old_pos, new_pos));
        old_pos += 1;
        new_pos += 1;
    }
    debug_assert_eq!(new_pos, new_lines.len());

    let mut hunks: Vec<Range<usize>> = Vec::with_capacity(change_ranges.len());
    for change in change_ranges {
        let start = change.start.saturating_sub(CONTEXT_LINES);
        let end = (change.end + CONTEXT_LINES).min(ops.len());
        match hunks.last_mut() {
            Some(last) if start <= last.end => last.end = end.max(last.end),
            _ => hunks.push(start..end),
        }
    }

    (ops, hunks, additions, deletions)
}

/// Emit side-by-side rows for a hunk: context lines occupy both halves,
/// runs of removals/additions are paired up like GitHub's split view.
fn push_split_rows(rows: &mut Vec<Row>, file_idx: usize, file: &FileDiff, hunk: Range<usize>) {
    let ops = &file.ops;
    let mut i = hunk.start;
    while i < hunk.end {
        match ops[i].kind {
            OpKind::Context => {
                rows.push(Row::SplitLine {
                    file: file_idx,
                    left: i,
                    right: i,
                });
                i += 1;
            }
            OpKind::Removed | OpKind::Added => {
                let removed_start = i;
                while i < hunk.end && ops[i].kind == OpKind::Removed {
                    i += 1;
                }
                let added_start = i;
                while i < hunk.end && ops[i].kind == OpKind::Added {
                    i += 1;
                }
                let removed = added_start - removed_start;
                let added = i - added_start;
                for k in 0..removed.max(added) {
                    rows.push(Row::SplitLine {
                        file: file_idx,
                        left: if k < removed {
                            removed_start + k
                        } else {
                            NO_LINE
                        },
                        right: if k < added { added_start + k } else { NO_LINE },
                    });
                }
            }
        }
    }
}

fn looks_binary(bytes: &[u8]) -> bool {
    bytes[..bytes.len().min(8192)].contains(&0)
}

fn display_path(path: &Path, cwd: &Path) -> String {
    path.strip_prefix(cwd).unwrap_or(path).display().to_string()
}

fn build_file_diff(
    registry: &DiffProviderRegistry,
    cwd: &Path,
    change: FileChange,
    trust_full: bool,
    ignore_whitespace: bool,
    rev: Option<&str>,
) -> FileDiff {
    let (status, old_path, path) = match change {
        FileChange::Untracked { path } => (FileStatus::Added, None, path),
        FileChange::Modified { path } => (FileStatus::Modified, None, path),
        FileChange::Conflict { path } => (FileStatus::Conflict, None, path),
        FileChange::Deleted { path } => (FileStatus::Deleted, None, path),
        FileChange::Renamed { from_path, to_path } => {
            (FileStatus::Renamed, Some(from_path), to_path)
        }
    };

    let base_path = old_path.as_deref().unwrap_or(&path);
    let old_bytes = match (status, rev) {
        (FileStatus::Added, _) => Vec::new(),
        (_, Some(rev)) => registry
            .get_diff_base_at(base_path, rev, trust_full)
            .unwrap_or_default(),
        (_, None) => registry
            .get_diff_base(base_path, trust_full)
            .unwrap_or_default(),
    };
    let new_bytes = match status {
        FileStatus::Deleted => Vec::new(),
        _ => std::fs::read(&path).unwrap_or_default(),
    };

    let binary = looks_binary(&old_bytes) || looks_binary(&new_bytes);
    let (old_lines, new_lines, ops, hunks, additions, deletions) = if binary {
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0, 0)
    } else {
        let old_lines = split_lines(&String::from_utf8_lossy(&old_bytes));
        let new_lines = split_lines(&String::from_utf8_lossy(&new_bytes));
        let (ops, hunks, additions, deletions) =
            compute_ops(&old_lines, &new_lines, ignore_whitespace);
        (old_lines, new_lines, ops, hunks, additions, deletions)
    };

    FileDiff {
        display_path: display_path(&path, cwd),
        old_display_path: old_path.as_deref().map(|p| display_path(p, cwd)),
        abs_path: path,
        status,
        binary,
        old_lines,
        new_lines,
        ops,
        hunks,
        additions,
        deletions,
        collapsed: additions + deletions > AUTO_COLLAPSE_CHANGES,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Row {
    FileHeader {
        file: usize,
    },
    HunkHeader {
        file: usize,
        hunk: usize,
    },
    /// Unified view line.
    Line {
        file: usize,
        op: usize,
    },
    /// Side-by-side view line; `NO_LINE` marks an empty half.
    SplitLine {
        file: usize,
        left: usize,
        right: usize,
    },
    /// Expander for hidden lines below the last hunk of a file.
    ExpandTail {
        file: usize,
    },
    Note {
        file: usize,
        text: &'static str,
    },
}

impl Row {
    fn file(self) -> usize {
        match self {
            Row::FileHeader { file }
            | Row::HunkHeader { file, .. }
            | Row::Line { file, .. }
            | Row::SplitLine { file, .. }
            | Row::ExpandTail { file }
            | Row::Note { file, .. } => file,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum GotoKind {
    Definition,
    References,
    TypeDefinition,
}

pub struct DiffViewer {
    files: Vec<FileDiff>,
    rows: Vec<Row>,
    cursor: usize,
    /// Char offset within the current diff line (for LSP jumps).
    cursor_col: usize,
    scroll: usize,
    h_scroll: usize,
    split: bool,
    ignore_whitespace: bool,
    /// Base revision the diff is computed against (`HEAD` when `None`).
    rev: Option<String>,
    show_help: bool,
    pending: Option<char>,
    /// Line-number gutter width, derived from the largest file.
    num_width: usize,
    /// The rows viewport of the last render, for mouse hit-testing.
    viewport: Rect,
}

impl DiffViewer {
    pub const ID: &'static str = "diff-viewer";

    pub fn new(
        files: Vec<FileDiff>,
        split: bool,
        ignore_whitespace: bool,
        rev: Option<String>,
    ) -> Self {
        let max_line = files
            .iter()
            .map(|f| f.old_lines.len().max(f.new_lines.len()))
            .max()
            .unwrap_or(0);
        let num_width = max_line.max(1).ilog10() as usize + 1;
        let mut viewer = Self {
            files,
            rows: Vec::new(),
            cursor: 0,
            cursor_col: 0,
            scroll: 0,
            h_scroll: 0,
            split,
            ignore_whitespace,
            rev,
            show_help: false,
            pending: None,
            num_width: num_width.max(3),
            viewport: Rect::default(),
        };
        viewer.rebuild_rows();
        viewer
    }

    fn rebuild_rows(&mut self) {
        let mut rows = Vec::new();
        for (file_idx, file) in self.files.iter().enumerate() {
            rows.push(Row::FileHeader { file: file_idx });
            if file.collapsed {
                continue;
            }
            if file.binary {
                rows.push(Row::Note {
                    file: file_idx,
                    text: "binary file (contents not shown)",
                });
                continue;
            }
            if file.hunks.is_empty() {
                let text = match file.status {
                    FileStatus::Renamed => "file renamed without content changes",
                    _ => "no content changes",
                };
                rows.push(Row::Note {
                    file: file_idx,
                    text,
                });
                continue;
            }
            for (hunk_idx, hunk) in file.hunks.iter().enumerate() {
                rows.push(Row::HunkHeader {
                    file: file_idx,
                    hunk: hunk_idx,
                });
                if self.split {
                    push_split_rows(&mut rows, file_idx, file, hunk.clone());
                } else {
                    for op in hunk.clone() {
                        rows.push(Row::Line { file: file_idx, op });
                    }
                }
            }
            if file.gap_below() > 0 {
                rows.push(Row::ExpandTail { file: file_idx });
            }
        }
        self.rows = rows;
        self.cursor = self.cursor.min(self.rows.len().saturating_sub(1));
    }

    fn recompute_diffs(&mut self) {
        for file in &mut self.files {
            if file.binary {
                continue;
            }
            let (ops, hunks, additions, deletions) =
                compute_ops(&file.old_lines, &file.new_lines, self.ignore_whitespace);
            file.ops = ops;
            file.hunks = hunks;
            file.additions = additions;
            file.deletions = deletions;
        }
        self.rebuild_rows();
    }

    fn current_row(&self) -> Option<Row> {
        self.rows.get(self.cursor).copied()
    }

    fn move_cursor(&mut self, delta: isize) {
        if self.rows.is_empty() {
            return;
        }
        let max = self.rows.len() as isize - 1;
        self.cursor = (self.cursor as isize + delta).clamp(0, max) as usize;
    }

    fn goto_row_matching(&mut self, forward: bool, pred: impl Fn(Row) -> bool) {
        let found = if forward {
            self.rows
                .iter()
                .enumerate()
                .skip(self.cursor + 1)
                .find(|(_, row)| pred(**row))
                .map(|(idx, _)| idx)
        } else {
            self.rows[..self.cursor]
                .iter()
                .enumerate()
                .rev()
                .find(|(_, row)| pred(**row))
                .map(|(idx, _)| idx)
        };
        if let Some(idx) = found {
            self.cursor = idx;
        }
    }

    fn next_file(&mut self, forward: bool) {
        self.goto_row_matching(forward, |row| matches!(row, Row::FileHeader { .. }));
    }

    fn next_hunk(&mut self, forward: bool) {
        self.goto_row_matching(forward, |row| matches!(row, Row::HunkHeader { .. }));
    }

    fn toggle_collapse(&mut self, file_idx: usize) {
        self.files[file_idx].collapsed = !self.files[file_idx].collapsed;
        self.rebuild_rows();
        // Keep the cursor on the toggled file's header.
        if let Some(idx) = self
            .rows
            .iter()
            .position(|row| *row == Row::FileHeader { file: file_idx })
        {
            self.cursor = idx;
        }
    }

    fn set_all_collapsed(&mut self, collapsed: bool) {
        let current_file = self.current_row().map(|row| row.file());
        for file in &mut self.files {
            file.collapsed = collapsed;
        }
        self.rebuild_rows();
        if let Some(file) = current_file {
            if let Some(idx) = self
                .rows
                .iter()
                .position(|row| *row == Row::FileHeader { file })
            {
                self.cursor = idx;
            }
        }
    }

    /// Reveal hidden context above the given hunk. Expands by
    /// `EXPAND_LINES` or, with `all`, the entire gap; merges with the
    /// previous hunk when they meet.
    fn expand_above(&mut self, file_idx: usize, hunk_idx: usize, all: bool) {
        let file = &mut self.files[file_idx];
        let prev_end = if hunk_idx == 0 {
            0
        } else {
            file.hunks[hunk_idx - 1].end
        };
        let hunk = &mut file.hunks[hunk_idx];
        hunk.start = if all {
            prev_end
        } else {
            hunk.start.saturating_sub(EXPAND_LINES).max(prev_end)
        };
        let mut target_hunk = hunk_idx;
        if hunk_idx > 0 && file.hunks[hunk_idx].start <= file.hunks[hunk_idx - 1].end {
            let merged = file.hunks.remove(hunk_idx);
            file.hunks[hunk_idx - 1].end = merged.end.max(file.hunks[hunk_idx - 1].end);
            target_hunk = hunk_idx - 1;
        }
        self.rebuild_rows();
        if let Some(idx) = self.rows.iter().position(|row| {
            *row == Row::HunkHeader {
                file: file_idx,
                hunk: target_hunk,
            }
        }) {
            self.cursor = idx;
        }
    }

    /// Reveal hidden lines below the last hunk of a file.
    fn expand_tail(&mut self, file_idx: usize, all: bool) {
        let file = &mut self.files[file_idx];
        let len = file.ops.len();
        if let Some(hunk) = file.hunks.last_mut() {
            hunk.end = if all {
                len
            } else {
                (hunk.end + EXPAND_LINES).min(len)
            };
        }
        self.rebuild_rows();
    }

    fn expand_at_cursor(&mut self, all: bool) {
        match self.current_row() {
            Some(Row::HunkHeader { file, hunk }) => self.expand_above(file, hunk, all),
            Some(Row::ExpandTail { file }) => self.expand_tail(file, all),
            _ => (),
        }
    }

    /// The file and (optionally) the op the cursor is on.
    fn target_at_cursor(&self) -> Option<(usize, Option<usize>)> {
        match self.current_row()? {
            Row::Line { file, op } => Some((file, Some(op))),
            Row::SplitLine { file, left, right } => {
                let op = if right != NO_LINE { right } else { left };
                Some((file, (op != NO_LINE).then_some(op)))
            }
            Row::FileHeader { file }
            | Row::HunkHeader { file, .. }
            | Row::ExpandTail { file }
            | Row::Note { file, .. } => Some((file, None)),
        }
    }

    /// Target (line, column) in the *current* file for a jump from the
    /// cursor. Removed lines map to the position where the removal
    /// happened (the next surviving line).
    fn jump_position(&self, file: &FileDiff, op_idx: Option<usize>) -> (usize, usize) {
        let line = op_idx
            .and_then(|idx| {
                file.ops[idx..]
                    .iter()
                    .find(|op| op.new != NO_LINE)
                    .map(|op| op.new)
            })
            .unwrap_or(0);
        (line, self.cursor_col)
    }

    /// The text of the line under the cursor, if the cursor is on one.
    fn line_under_cursor(&self) -> Option<&str> {
        let (file_idx, op_idx) = self.target_at_cursor()?;
        let file = &self.files[file_idx];
        let op = &file.ops[op_idx?];
        let line = if op.new != NO_LINE {
            file.new_lines.get(op.new)
        } else {
            file.old_lines.get(op.old)
        };
        line.map(String::as_str)
    }

    fn clamp_cursor_col(&mut self) {
        let max = self
            .line_under_cursor()
            .map(|line| line.chars().count())
            .unwrap_or(0);
        self.cursor_col = self.cursor_col.min(max.saturating_sub(1).max(0));
        if max == 0 {
            self.cursor_col = 0;
        }
    }

    fn move_col(&mut self, delta: isize) {
        let max = self
            .line_under_cursor()
            .map(|line| line.chars().count())
            .unwrap_or(0);
        let max = max.saturating_sub(1) as isize;
        self.cursor_col = (self.cursor_col as isize + delta).clamp(0, max.max(0)) as usize;
    }

    fn move_word(&mut self, forward: bool) {
        let Some(line) = self.line_under_cursor() else {
            return;
        };
        let chars: Vec<char> = line.chars().collect();
        if chars.is_empty() {
            return;
        }
        let is_word = |c: char| c.is_alphanumeric() || c == '_';
        let mut idx = self.cursor_col.min(chars.len() - 1);
        if forward {
            // Skip the rest of the current word, then whitespace/punctuation.
            while idx + 1 < chars.len() && is_word(chars[idx]) && is_word(chars[idx + 1]) {
                idx += 1;
            }
            idx = (idx + 1).min(chars.len() - 1);
            while idx + 1 < chars.len() && !is_word(chars[idx]) {
                idx += 1;
            }
        } else {
            idx = idx.saturating_sub(1);
            while idx > 0 && !is_word(chars[idx]) {
                idx -= 1;
            }
            while idx > 0 && is_word(chars[idx - 1]) {
                idx -= 1;
            }
        }
        self.cursor_col = idx;
    }

    fn first_non_whitespace_col(&self) -> usize {
        self.line_under_cursor()
            .map(|line| line.chars().position(|c| !c.is_whitespace()).unwrap_or(0))
            .unwrap_or(0)
    }

    fn close() -> EventResult {
        EventResult::Consumed(Some(Box::new(|compositor: &mut Compositor, _| {
            compositor.pop();
        })))
    }

    fn reload(&self, cx: &mut Context) -> EventResult {
        open_with(
            cx.editor,
            cx.jobs,
            self.split,
            self.ignore_whitespace,
            self.rev.clone(),
        );
        EventResult::Consumed(None)
    }

    /// Open the file under the cursor at the corresponding position,
    /// closing the viewer. With `goto`, additionally trigger the LSP
    /// goto-* command at that position.
    fn open_at_cursor(&self, action: Action, goto: Option<GotoKind>) -> EventResult {
        let Some((file_idx, op_idx)) = self.target_at_cursor() else {
            return EventResult::Consumed(None);
        };
        let file = &self.files[file_idx];
        if file.status == FileStatus::Deleted {
            return EventResult::Consumed(Some(Box::new(|_, cx: &mut Context| {
                cx.editor
                    .set_error("cannot open a deleted file; expand the diff to view it");
            })));
        }
        let path = file.abs_path.clone();
        let (line, col) = self.jump_position(file, op_idx);

        EventResult::Consumed(Some(Box::new(
            move |compositor: &mut Compositor, cx: &mut Context| {
                compositor.pop();
                // Record the jump origin so `C-o` returns here.
                {
                    let (view, doc) = helix_view::current!(cx.editor);
                    let jump = (doc.id(), doc.selection(view.id).clone());
                    view.push_jump(doc, jump);
                }
                if let Err(err) = cx.editor.open(&path, action) {
                    cx.editor
                        .set_error(format!("unable to open \"{}\": {}", path.display(), err));
                    return;
                }
                let (view, doc) = helix_view::current!(cx.editor);
                let text = doc.text();
                let line = line.min(text.len_lines().saturating_sub(1));
                let line_len = text.line(line).len_chars();
                let pos = text.line_to_char(line) + col.min(line_len.saturating_sub(1));
                doc.set_selection(view.id, Selection::point(pos));
                align_view(doc, view, Align::Center);

                if let Some(goto) = goto {
                    let mut cmd_cx = crate::commands::Context {
                        register: None,
                        count: None,
                        editor: cx.editor,
                        callback: Vec::new(),
                        on_next_key_callback: None,
                        jobs: cx.jobs,
                    };
                    match goto {
                        GotoKind::Definition => crate::commands::lsp::goto_definition(&mut cmd_cx),
                        GotoKind::References => crate::commands::lsp::goto_reference(&mut cmd_cx),
                        GotoKind::TypeDefinition => {
                            crate::commands::lsp::goto_type_definition(&mut cmd_cx)
                        }
                    }
                    let callbacks = cmd_cx.callback;
                    for callback in callbacks {
                        callback(compositor, cx);
                    }
                }
            },
        )))
    }

    fn handle_mouse(&mut self, event: &MouseEvent) -> EventResult {
        match event.kind {
            MouseEventKind::ScrollUp => {
                self.scroll = self.scroll.saturating_sub(3);
                self.cursor = self
                    .cursor
                    .min(self.scroll + self.viewport.height.saturating_sub(1) as usize);
                self.clamp_cursor_col();
            }
            MouseEventKind::ScrollDown => {
                let max = self.rows.len().saturating_sub(1);
                self.scroll = (self.scroll + 3).min(max);
                self.cursor = self.cursor.max(self.scroll);
                self.clamp_cursor_col();
            }
            MouseEventKind::Down(MouseButton::Left) => {
                let viewport = self.viewport;
                if event.row < viewport.y || event.row >= viewport.y + viewport.height {
                    return EventResult::Consumed(None);
                }
                let idx = self.scroll + (event.row - viewport.y) as usize;
                if idx >= self.rows.len() {
                    return EventResult::Consumed(None);
                }
                self.cursor = idx;
                self.clamp_cursor_col();
                match self.rows[idx] {
                    Row::FileHeader { file } => self.toggle_collapse(file),
                    Row::HunkHeader { .. } | Row::ExpandTail { .. } => self.expand_at_cursor(false),
                    _ => (),
                }
            }
            _ => (),
        }
        EventResult::Consumed(None)
    }
}

pub fn open(editor: &mut Editor, jobs: &mut Jobs, rev: Option<String>) {
    open_with(editor, jobs, false, false, rev);
}

fn open_with(
    editor: &mut Editor,
    jobs: &mut Jobs,
    split: bool,
    ignore_whitespace: bool,
    rev: Option<String>,
) {
    let cwd = helix_stdx::env::current_working_dir();
    if !cwd.exists() {
        editor.set_error("current working directory does not exist");
        return;
    }
    let trust_full = editor
        .workspace_trust
        .query(
            &helix_loader::find_workspace_in(&cwd).0,
            helix_loader::workspace_trust::TrustQuery::Git,
        )
        .is_trusted();
    let registry = editor.diff_providers.clone();
    jobs.callback(gather_diffs(
        registry,
        cwd,
        trust_full,
        split,
        ignore_whitespace,
        rev,
    ));
}

async fn gather_diffs(
    registry: DiffProviderRegistry,
    cwd: PathBuf,
    trust_full: bool,
    split: bool,
    ignore_whitespace: bool,
    rev: Option<String>,
) -> Result<job::Callback> {
    let scope = match &rev {
        Some(rev) => StatusScope::MergeBaseToWorktree(rev.clone()),
        None => StatusScope::HeadToWorktree,
    };
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    registry
        .clone()
        .for_each_changed_file(cwd.clone(), trust_full, scope, move |change| {
            tx.send(change).is_ok()
        });

    let mut changes = Vec::new();
    let mut error: Option<String> = None;
    while let Some(change) = rx.recv().await {
        match change {
            Ok(change) => changes.push(change),
            Err(err) => {
                error.get_or_insert_with(|| err.to_string());
            }
        }
    }

    let files = {
        let cwd = cwd.clone();
        let rev = rev.clone();
        tokio::task::spawn_blocking(move || {
            let mut files: Vec<FileDiff> = changes
                .into_iter()
                .map(|change| {
                    build_file_diff(
                        &registry,
                        &cwd,
                        change,
                        trust_full,
                        ignore_whitespace,
                        rev.as_deref(),
                    )
                })
                .collect();
            // A file changed both in the index and the worktree is reported
            // twice; keep one entry per path, preferring the one carrying
            // rename information.
            files.sort_by(|a, b| {
                a.abs_path.cmp(&b.abs_path).then_with(|| {
                    a.old_display_path
                        .is_none()
                        .cmp(&b.old_display_path.is_none())
                })
            });
            files.dedup_by(|a, b| a.abs_path == b.abs_path);
            files
        })
        .await?
    };

    Ok(job::Callback::EditorCompositor(Box::new(
        move |editor: &mut Editor, compositor: &mut Compositor| {
            if files.is_empty() {
                match error {
                    Some(err) => editor.set_error(format!("diff: {}", err)),
                    None => editor.set_status("diff: no changes in workspace"),
                }
                return;
            }
            compositor.replace_or_push(
                DiffViewer::ID,
                DiffViewer::new(files, split, ignore_whitespace, rev),
            );
        },
    )))
}

// Rendering ------------------------------------------------------------------

struct Styles {
    text: Style,
    dim: Style,
    linenr: Style,
    plus: Style,
    minus: Style,
    file_header: Style,
    hunk_header: Style,
    statusline: Style,
    cursorline: Style,
    cursor: Style,
    separator: Style,
}

impl Styles {
    fn new(theme: &Theme) -> Self {
        Styles {
            text: theme.get("ui.text"),
            dim: theme
                .try_get("ui.text.inactive")
                .unwrap_or_else(|| theme.get("comment")),
            linenr: theme.get("ui.linenr"),
            plus: theme.get("diff.plus"),
            minus: theme.get("diff.minus"),
            file_header: theme.get("ui.statusline").add_modifier(Modifier::BOLD),
            hunk_header: theme.get("diff.delta").add_modifier(Modifier::DIM),
            statusline: theme.get("ui.statusline"),
            cursorline: theme.get("ui.cursorline.primary"),
            cursor: theme
                .try_get_exact("ui.cursor")
                .unwrap_or_else(|| Style::default().add_modifier(Modifier::REVERSED)),
            separator: theme.get("ui.window"),
        }
    }

    fn status(&self, theme: &Theme, status: FileStatus) -> Style {
        theme.get(status.theme_scope()).add_modifier(Modifier::BOLD)
    }
}

fn char_width(ch: char, col: usize) -> usize {
    match ch {
        '\t' => 4 - col % 4,
        _ => UnicodeWidthChar::width(ch).unwrap_or(1).max(1),
    }
}

fn visual_col(line: &str, char_idx: usize) -> usize {
    let mut col = 0;
    for (idx, ch) in line.chars().enumerate() {
        if idx >= char_idx {
            break;
        }
        col += char_width(ch, col);
    }
    col
}

/// Draw one line of code at `(x, y)`, horizontally scrolled by `skip`
/// visual columns, applying `hl_style` to the `hl` byte range.
#[allow(clippy::too_many_arguments)]
fn draw_code_line(
    surface: &mut Surface,
    x: u16,
    y: u16,
    width: usize,
    line: &str,
    base: Style,
    hl: Option<&Range<usize>>,
    hl_style: Style,
    skip: usize,
) {
    let mut vcol = 0usize;
    let mut byte = 0usize;
    let mut char_buf = [0u8; 4];
    for ch in line.chars() {
        let w = char_width(ch, vcol);
        let start = vcol;
        let end = vcol + w;
        vcol = end;
        let ch_byte = byte;
        byte += ch.len_utf8();
        if end <= skip {
            continue;
        }
        if start >= skip + width {
            break;
        }
        let style = if hl.is_some_and(|r| r.contains(&ch_byte)) {
            hl_style
        } else {
            base
        };
        if ch == '\t' || start < skip {
            // Expand tabs; pad wide chars clipped at the left edge.
            for v in start.max(skip)..end.min(skip + width) {
                surface.set_stringn(x + (v - skip) as u16, y, " ", 1, style);
            }
        } else {
            let s: &str = ch.encode_utf8(&mut char_buf);
            let remaining = skip + width - start;
            surface.set_stringn(x + (start - skip) as u16, y, s, remaining, style);
        }
    }
}

impl Component for DiffViewer {
    fn render(&mut self, area: Rect, surface: &mut Surface, cx: &mut Context) {
        let theme = &cx.editor.theme;
        let styles = Styles::new(theme);
        surface.clear_with(area, styles.text);

        if area.height < 4 || area.width < 20 {
            return;
        }

        // Header -------------------------------------------------------
        let header_area = area.with_height(1);
        surface.clear_with(header_area, styles.statusline);
        let (total_add, total_del) = self
            .files
            .iter()
            .fold((0, 0), |(a, d), f| (a + f.additions, d + f.deletions));
        let base_label = match &self.rev {
            Some(rev) => format!("{rev}...working tree"),
            None => "HEAD → working tree".to_string(),
        };
        let header = format!(
            " {} changed file{} with {} additions and {} deletions • {} • {} view{}",
            self.files.len(),
            if self.files.len() == 1 { "" } else { "s" },
            total_add,
            total_del,
            base_label,
            if self.split { "split" } else { "unified" },
            if self.ignore_whitespace {
                " • ignoring whitespace"
            } else {
                ""
            },
        );
        surface.set_stringn(
            header_area.x,
            header_area.y,
            &header,
            header_area.width.saturating_sub(8) as usize,
            styles.statusline,
        );
        let help_hint = "? help ";
        if (header_area.width as usize) > help_hint.len() {
            surface.set_string(
                header_area.right() - help_hint.len() as u16,
                header_area.y,
                help_hint,
                styles.statusline,
            );
        }

        // Footer -------------------------------------------------------
        let footer_area = Rect::new(area.x, area.bottom() - 1, area.width, 1);
        surface.clear_with(footer_area, styles.statusline);
        let footer = " enter open  gd/gr/gy goto  n/p hunk  tab/S-tab file  za fold  +/= expand  s split  W whitespace  r reload  q quit";
        surface.set_stringn(
            footer_area.x,
            footer_area.y,
            footer,
            footer_area.width as usize,
            styles.statusline,
        );

        // Diff rows ----------------------------------------------------
        let inner = area.clip_top(1).clip_bottom(1);
        self.viewport = inner;
        let height = inner.height as usize;

        // Keep the cursor row in view.
        self.cursor = self.cursor.min(self.rows.len().saturating_sub(1));
        if self.cursor < self.scroll {
            self.scroll = self.cursor;
        } else if self.cursor >= self.scroll + height {
            self.scroll = self.cursor + 1 - height;
        }
        self.scroll = self.scroll.min(self.rows.len().saturating_sub(1));

        // Keep the cursor column in view (horizontal scroll).
        let text_width = self.text_width(inner.width);
        if let Some(line) = self.line_under_cursor() {
            let vcol = visual_col(line, self.cursor_col.min(line.chars().count()));
            if vcol < self.h_scroll {
                self.h_scroll = vcol;
            } else if vcol >= self.h_scroll + text_width {
                self.h_scroll = vcol + 1 - text_width;
            }
        }

        for (i, row) in self.rows.iter().skip(self.scroll).take(height).enumerate() {
            let y = inner.y + i as u16;
            let row_idx = self.scroll + i;
            self.draw_row(surface, inner, y, *row, theme, &styles);
            if row_idx == self.cursor {
                surface.set_style(Rect::new(inner.x, y, inner.width, 1), styles.cursorline);
                self.draw_cell_cursor(surface, inner, y, *row, &styles);
            }
        }

        // Scrollbar ------------------------------------------------------
        if self.rows.len() > height {
            let scrollbar_x = inner.right().saturating_sub(1);
            let thumb_height = ((height * height) / self.rows.len()).max(1);
            let thumb_top = (self.scroll * height.saturating_sub(thumb_height))
                / (self.rows.len() - height).max(1);
            for i in 0..thumb_height.min(height) {
                surface.set_stringn(
                    scrollbar_x,
                    inner.y + (thumb_top + i) as u16,
                    "▐",
                    1,
                    styles.dim,
                );
            }
        }

        if self.show_help {
            self.draw_help(surface, area, theme, &styles);
        }
    }

    fn handle_event(&mut self, event: &Event, cx: &mut Context) -> EventResult {
        let mut key = match event {
            Event::Key(key) => *key,
            Event::Mouse(mouse) => return self.handle_mouse(mouse),
            Event::Resize(..) => return EventResult::Consumed(None),
            _ => return EventResult::Ignored(None),
        };
        // Uppercase chars may arrive with an explicit SHIFT modifier
        // depending on the terminal; normalize like the editor view does.
        if matches!(key.code, helix_view::keyboard::KeyCode::Char(_)) {
            key.modifiers
                .remove(helix_view::keyboard::KeyModifiers::SHIFT);
        }

        if self.show_help {
            self.show_help = false;
            return EventResult::Consumed(None);
        }

        if let Some(pending) = self.pending.take() {
            match (pending, key) {
                ('g', key!('g')) => {
                    self.cursor = 0;
                    self.clamp_cursor_col();
                }
                ('g', key!('e')) => {
                    self.cursor = self.rows.len().saturating_sub(1);
                    self.clamp_cursor_col();
                }
                ('g', key!('h')) => self.cursor_col = 0,
                ('g', key!('l')) => self.move_col(isize::MAX / 2),
                ('g', key!('s')) => self.cursor_col = self.first_non_whitespace_col(),
                ('g', key!('d')) => {
                    return self.open_at_cursor(Action::Replace, Some(GotoKind::Definition))
                }
                ('g', key!('r')) => {
                    return self.open_at_cursor(Action::Replace, Some(GotoKind::References))
                }
                ('g', key!('y')) => {
                    return self.open_at_cursor(Action::Replace, Some(GotoKind::TypeDefinition))
                }
                ('z', key!('a')) => {
                    if let Some(row) = self.current_row() {
                        self.toggle_collapse(row.file());
                    }
                }
                ('z', key!('M')) => self.set_all_collapsed(true),
                ('z', key!('R')) => self.set_all_collapsed(false),
                ('z', key!('z')) => {
                    let height = self.viewport.height as usize;
                    self.scroll = self.cursor.saturating_sub(height / 2);
                }
                (']', key!('c')) => self.next_hunk(true),
                ('[', key!('c')) => self.next_hunk(false),
                (']', key!('f')) => self.next_file(true),
                ('[', key!('f')) => self.next_file(false),
                _ => (),
            }
            return EventResult::Consumed(None);
        }

        match key {
            key!('q') | ctrl!('c') => return Self::close(),
            key!(Esc) => return Self::close(),
            key!('?') => self.show_help = true,

            key!('j') | key!(Down) => {
                self.move_cursor(1);
                self.clamp_cursor_col();
            }
            key!('k') | key!(Up) => {
                self.move_cursor(-1);
                self.clamp_cursor_col();
            }
            key!('h') | key!(Left) => self.move_col(-1),
            key!('l') | key!(Right) => self.move_col(1),
            key!('w') => self.move_word(true),
            key!('b') => self.move_word(false),
            ctrl!('d') | key!(PageDown) => {
                let page = (self.viewport.height as isize / 2).max(1);
                self.move_cursor(page);
                self.clamp_cursor_col();
            }
            ctrl!('u') | key!(PageUp) => {
                let page = (self.viewport.height as isize / 2).max(1);
                self.move_cursor(-page);
                self.clamp_cursor_col();
            }
            key!('G') | key!(End) => {
                self.cursor = self.rows.len().saturating_sub(1);
                self.clamp_cursor_col();
            }
            key!(Home) => {
                self.cursor = 0;
                self.clamp_cursor_col();
            }

            key!('g') => self.pending = Some('g'),
            key!('z') => self.pending = Some('z'),
            key!(']') => self.pending = Some(']'),
            key!('[') => self.pending = Some('['),

            key!(Tab) => self.next_file(true),
            shift!(Tab) => self.next_file(false),
            key!('n') => self.next_hunk(true),
            key!('p') => self.next_hunk(false),

            key!('+') => self.expand_at_cursor(false),
            key!('=') => self.expand_at_cursor(true),

            key!('s') => {
                self.split = !self.split;
                self.rebuild_rows();
            }
            key!('W') => {
                self.ignore_whitespace = !self.ignore_whitespace;
                self.recompute_diffs();
            }
            key!('r') => return self.reload(cx),

            key!(Enter) => match self.current_row() {
                Some(Row::FileHeader { file }) => self.toggle_collapse(file),
                Some(Row::HunkHeader { .. }) | Some(Row::ExpandTail { .. }) => {
                    self.expand_at_cursor(false)
                }
                Some(_) => return self.open_at_cursor(Action::Replace, None),
                None => (),
            },
            ctrl!('s') => return self.open_at_cursor(Action::HorizontalSplit, None),
            ctrl!('v') => return self.open_at_cursor(Action::VerticalSplit, None),

            _ => (),
        }
        EventResult::Consumed(None)
    }

    fn id(&self) -> Option<&'static str> {
        Some(Self::ID)
    }
}

impl DiffViewer {
    /// Width available for code text (per pane in split view).
    fn text_width(&self, total: u16) -> usize {
        let total = total as usize;
        if self.split {
            let pane = total.saturating_sub(1) / 2;
            pane.saturating_sub(self.num_width + 3).max(10)
        } else {
            total.saturating_sub(2 * self.num_width + 4).max(10)
        }
    }

    fn draw_row(
        &self,
        surface: &mut Surface,
        inner: Rect,
        y: u16,
        row: Row,
        theme: &Theme,
        styles: &Styles,
    ) {
        match row {
            Row::FileHeader { file } => {
                self.draw_file_header(surface, inner, y, file, theme, styles)
            }
            Row::HunkHeader { file, hunk } => {
                let f = &self.files[file];
                let (old_start, old_count, new_start, new_count) = f.hunk_line_info(&f.hunks[hunk]);
                let gap = f.gap_above(hunk);
                let mut text = format!(
                    "  @@ -{},{} +{},{} @@",
                    old_start, old_count, new_start, new_count
                );
                if gap > 0 {
                    text.push_str(&format!("  ⋯ {} hidden lines (+ expand, = all)", gap));
                }
                surface.set_stringn(inner.x, y, &text, inner.width as usize, styles.hunk_header);
            }
            Row::Line { file, op } => self.draw_unified_line(surface, inner, y, file, op, styles),
            Row::SplitLine { file, left, right } => {
                self.draw_split_line(surface, inner, y, file, left, right, styles)
            }
            Row::ExpandTail { file } => {
                let gap = self.files[file].gap_below();
                let text = format!("  ⋯ {} hidden lines below (+ expand, = all)", gap);
                surface.set_stringn(inner.x, y, &text, inner.width as usize, styles.hunk_header);
            }
            Row::Note { file: _, text } => {
                surface.set_stringn(
                    inner.x + 2,
                    y,
                    text,
                    inner.width.saturating_sub(2) as usize,
                    styles.dim,
                );
            }
        }
    }

    fn draw_file_header(
        &self,
        surface: &mut Surface,
        inner: Rect,
        y: u16,
        file_idx: usize,
        theme: &Theme,
        styles: &Styles,
    ) {
        let file = &self.files[file_idx];
        let header_area = Rect::new(inner.x, y, inner.width, 1);
        surface.clear_with(header_area, styles.file_header);

        let arrow = if file.collapsed { "▸" } else { "▾" };
        let mut x = inner.x;
        let (end_x, _) = surface.set_stringn(
            x,
            y,
            &format!(" {} ", arrow),
            inner.width as usize,
            styles.file_header,
        );
        x = end_x;
        let (end_x, _) = surface.set_stringn(
            x,
            y,
            file.status.letter(),
            (inner.right().saturating_sub(x)) as usize,
            styles.status(theme, file.status).patch(styles.file_header),
        );
        x = end_x;
        let path_text = match &file.old_display_path {
            Some(old) => format!(" {} → {}", old, file.display_path),
            None => format!(" {}", file.display_path),
        };
        let (end_x, _) = surface.set_stringn(
            x,
            y,
            &path_text,
            (inner.right().saturating_sub(x)) as usize,
            styles.file_header,
        );
        x = end_x;
        if file.binary {
            surface.set_stringn(
                x,
                y,
                "  BIN",
                (inner.right().saturating_sub(x)) as usize,
                styles.file_header,
            );
        } else {
            let (end_x, _) = surface.set_stringn(
                x,
                y,
                &format!("  +{}", file.additions),
                (inner.right().saturating_sub(x)) as usize,
                styles.plus.patch(styles.file_header),
            );
            surface.set_stringn(
                end_x,
                y,
                &format!(" -{}", file.deletions),
                (inner.right().saturating_sub(end_x)) as usize,
                styles.minus.patch(styles.file_header),
            );
        }
    }

    fn line_style(&self, kind: OpKind, styles: &Styles) -> (char, Style) {
        match kind {
            OpKind::Context => (' ', styles.text),
            OpKind::Removed => ('-', styles.minus),
            OpKind::Added => ('+', styles.plus),
        }
    }

    fn draw_unified_line(
        &self,
        surface: &mut Surface,
        inner: Rect,
        y: u16,
        file_idx: usize,
        op_idx: usize,
        styles: &Styles,
    ) {
        let file = &self.files[file_idx];
        let op = &file.ops[op_idx];
        let (marker, style) = self.line_style(op.kind, styles);
        let w = self.num_width;

        let old_num = if op.old != NO_LINE {
            format!("{:>w$}", op.old + 1, w = w)
        } else {
            " ".repeat(w)
        };
        let new_num = if op.new != NO_LINE {
            format!("{:>w$}", op.new + 1, w = w)
        } else {
            " ".repeat(w)
        };
        let gutter = format!("{} {} {} ", old_num, new_num, marker);
        let (text_x, _) = surface.set_stringn(
            inner.x,
            y,
            &gutter,
            inner.width as usize,
            if op.kind == OpKind::Context {
                styles.linenr
            } else {
                style
            },
        );

        let line = if op.new != NO_LINE {
            file.new_lines.get(op.new)
        } else {
            file.old_lines.get(op.old)
        };
        if let Some(line) = line {
            let width = (inner.right().saturating_sub(text_x)) as usize;
            draw_code_line(
                surface,
                text_x,
                y,
                width,
                line,
                style,
                op.hl.as_ref(),
                style.add_modifier(Modifier::REVERSED),
                self.h_scroll,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_split_line(
        &self,
        surface: &mut Surface,
        inner: Rect,
        y: u16,
        file_idx: usize,
        left: usize,
        right: usize,
        styles: &Styles,
    ) {
        let file = &self.files[file_idx];
        let pane_width = (inner.width.saturating_sub(1) / 2) as usize;
        let sep_x = inner.x + pane_width as u16;
        surface.set_stringn(sep_x, y, "│", 1, styles.separator);

        let mut draw_half = |op_idx: usize, x: u16, width: usize, old_side: bool| {
            if op_idx == NO_LINE {
                return;
            }
            let op = &file.ops[op_idx];
            let (line_idx, lines) = if old_side {
                (op.old, &file.old_lines)
            } else {
                (op.new, &file.new_lines)
            };
            if line_idx == NO_LINE {
                return;
            }
            let (marker, style) = self.line_style(op.kind, styles);
            let gutter = format!("{:>w$} {} ", line_idx + 1, marker, w = self.num_width);
            let (text_x, _) = surface.set_stringn(
                x,
                y,
                &gutter,
                width,
                if op.kind == OpKind::Context {
                    styles.linenr
                } else {
                    style
                },
            );
            let text_width = width.saturating_sub((text_x - x) as usize);
            if let Some(line) = lines.get(line_idx) {
                draw_code_line(
                    surface,
                    text_x,
                    y,
                    text_width,
                    line,
                    style,
                    op.hl.as_ref(),
                    style.add_modifier(Modifier::REVERSED),
                    self.h_scroll,
                );
            }
        };

        draw_half(left, inner.x, pane_width, true);
        draw_half(right, sep_x + 1, pane_width, false);
    }

    /// Draw the cell cursor on the code line under the cursor row.
    fn draw_cell_cursor(
        &self,
        surface: &mut Surface,
        inner: Rect,
        y: u16,
        row: Row,
        styles: &Styles,
    ) {
        let on_line = matches!(row, Row::Line { .. } | Row::SplitLine { .. });
        if !on_line {
            return;
        }
        let Some(line) = self.line_under_cursor() else {
            return;
        };
        let col = self.cursor_col.min(line.chars().count());
        let vcol = visual_col(line, col);
        if vcol < self.h_scroll {
            return;
        }
        // Mirror the gutter layout of draw_unified_line / draw_split_line.
        let text_x = if self.split {
            let pane_width = inner.width.saturating_sub(1) / 2;
            // The cursor targets the new side when present, otherwise the old side.
            let right_has_line = matches!(row, Row::SplitLine { right, .. } if right != NO_LINE)
                || matches!(row, Row::Line { .. });
            let base = if right_has_line {
                inner.x + pane_width + 1
            } else {
                inner.x
            };
            base + self.num_width as u16 + 2
        } else {
            inner.x + 2 * self.num_width as u16 + 3
        };
        let x = text_x + (vcol - self.h_scroll) as u16;
        if x < inner.right() {
            surface.set_style(Rect::new(x, y, 1, 1), styles.cursor);
        }
    }

    fn draw_help(&self, surface: &mut Surface, area: Rect, theme: &Theme, styles: &Styles) {
        const HELP: &[(&str, &str)] = &[
            ("j/k, ↑/↓", "move up/down"),
            ("h/l, w/b", "move within line (word-wise: w/b)"),
            ("gh/gl/gs", "line start / line end / first non-blank"),
            ("gg/ge, G", "first / last row"),
            ("C-d/C-u", "half page down/up"),
            ("n/p, ]c/[c", "next/previous hunk"),
            ("Tab/S-Tab, ]f/[f", "next/previous file"),
            ("Enter", "open file at line (headers: fold/expand)"),
            ("C-s/C-v", "open in horizontal/vertical split"),
            ("gd/gr/gy", "goto definition/references/type-def"),
            ("za", "fold/unfold current file"),
            ("zM/zR", "fold/unfold all files"),
            ("zz", "center cursor row"),
            ("+/=", "expand hidden context (step / all)"),
            ("s", "toggle side-by-side view"),
            ("W", "toggle ignore whitespace"),
            ("r", "reload from disk"),
            ("q/Esc", "close"),
        ];
        let width = 58u16.min(area.width.saturating_sub(2));
        let height = (HELP.len() as u16 + 2).min(area.height.saturating_sub(2));
        let popup_area = Rect::new(
            area.x + (area.width - width) / 2,
            area.y + (area.height - height) / 2,
            width,
            height,
        );
        let popup_style = theme.get("ui.popup");
        surface.clear_with(popup_area, popup_style);
        let block = Block::bordered()
            .title(" diff viewer keys ")
            .border_style(popup_style);
        let popup_inner = block.inner(popup_area);
        block.render(popup_area, surface);
        for (i, (keys, desc)) in HELP.iter().enumerate() {
            if i as u16 >= popup_inner.height {
                break;
            }
            let y = popup_inner.y + i as u16;
            surface.set_stringn(
                popup_inner.x + 1,
                y,
                keys,
                popup_inner.width.saturating_sub(1) as usize,
                styles.text.patch(popup_style).add_modifier(Modifier::BOLD),
            );
            surface.set_stringn(
                popup_inner.x + 19,
                y,
                desc,
                popup_inner.width.saturating_sub(20) as usize,
                styles.text.patch(popup_style),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &str) -> Vec<String> {
        split_lines(text)
    }

    #[test]
    fn split_lines_handles_terminators() {
        assert_eq!(lines(""), Vec::<String>::new());
        assert_eq!(lines("a\nb\n"), vec!["a", "b"]);
        assert_eq!(lines("a\nb"), vec!["a", "b"]);
        assert_eq!(lines("a\r\nb\r\n"), vec!["a", "b"]);
        assert_eq!(lines("a\n\nb\n"), vec!["a", "", "b"]);
    }

    #[test]
    fn compute_ops_modification() {
        let old = lines("a\nb\nc\n");
        let new = lines("a\nB\nc\n");
        let (ops, hunks, additions, deletions) = compute_ops(&old, &new, false);
        assert_eq!(additions, 1);
        assert_eq!(deletions, 1);
        assert_eq!(ops.len(), 4); // a, -b, +B, c
        assert_eq!(ops[1].kind, OpKind::Removed);
        assert_eq!(ops[2].kind, OpKind::Added);
        // Context is limited to the file bounds.
        assert_eq!(hunks, vec![0..4]);
    }

    #[test]
    fn compute_ops_context_and_merging() {
        // Two changes 3 lines apart: their 3-line contexts overlap, so they
        // must merge into a single display hunk.
        let old = lines("1\n2\nX\n4\n5\n6\nY\n8\n9\n");
        let new = lines("1\n2\nx\n4\n5\n6\ny\n8\n9\n");
        let (ops, hunks, ..) = compute_ops(&old, &new, false);
        assert_eq!(hunks.len(), 1);
        assert_eq!(ops.len(), 11); // 9 lines + 2 extra for the two changes
    }

    #[test]
    fn compute_ops_separate_hunks() {
        let mut old_text = String::new();
        let mut new_text = String::new();
        for i in 0..30 {
            old_text.push_str(&format!("line{}\n", i));
            new_text.push_str(&format!(
                "{}\n",
                if i == 2 || i == 25 {
                    format!("changed{}", i)
                } else {
                    format!("line{}", i)
                }
            ));
        }
        let (_, hunks, additions, deletions) =
            compute_ops(&lines(&old_text), &lines(&new_text), false);
        assert_eq!(hunks.len(), 2);
        assert_eq!(additions, 2);
        assert_eq!(deletions, 2);
    }

    #[test]
    fn intraline_prefix_suffix() {
        let (old_hl, new_hl) = intraline_ranges("let foo = 1;", "let bar = 1;").unwrap();
        assert_eq!(&"let foo = 1;"[old_hl], "foo");
        assert_eq!(&"let bar = 1;"[new_hl], "bar");
        // Nothing in common: no highlight.
        assert!(intraline_ranges("abc", "xyz").is_none());
        // Identical: no highlight.
        assert!(intraline_ranges("same", "same").is_none());
    }

    #[test]
    fn intraline_multibyte_boundaries() {
        // Shared multi-byte prefix/suffix must not split char boundaries.
        let (old_hl, new_hl) = intraline_ranges("héllo wörld", "héllo Wörld").unwrap();
        assert!("héllo wörld".get(old_hl.clone()).is_some());
        assert!("héllo Wörld".get(new_hl.clone()).is_some());
    }

    #[test]
    fn ignore_whitespace_diff() {
        let old = lines("fn main() {\n    foo();\n}\n");
        let new = lines("fn main() {\n\tfoo();\n}\n");
        let (_, hunks, ..) = compute_ops(&old, &new, false);
        assert_eq!(hunks.len(), 1);
        let (_, hunks, additions, deletions) = compute_ops(&old, &new, true);
        assert!(hunks.is_empty());
        assert_eq!(additions, 0);
        assert_eq!(deletions, 0);
    }

    #[test]
    fn pure_addition_and_deletion() {
        let (ops, hunks, additions, deletions) = compute_ops(&[], &lines("a\nb\n"), false);
        assert_eq!((additions, deletions), (2, 0));
        assert_eq!(ops.len(), 2);
        assert_eq!(hunks, vec![0..2]);

        let (ops, hunks, additions, deletions) = compute_ops(&lines("a\nb\n"), &[], false);
        assert_eq!((additions, deletions), (0, 2));
        assert_eq!(ops.len(), 2);
        assert_eq!(hunks, vec![0..2]);
    }
}
