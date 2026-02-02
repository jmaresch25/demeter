#!/usr/bin/env bash
# =============================================================================
# gitter.sh
#
# Purpose:
#   Safely sync remote updates FIRST, then re-apply local uncommitted changes,
#   and finally (optionally) commit + push those local changes.
#
# High-level flow:
#   1) Safety checks (repo, branch attached, upstream set)
#   2) Show current local status (what files are modified/untracked)
#   3) Fetch remote refs (download remote state)
#   4) Pull with rebase + autostash (remote-first; stash local WIP automatically)
#   5) If conflicts during rebase, force "remote wins" and continue
#   6) Show final status
#   7) If there are changes, commit them and push to remote
# =============================================================================

set -euo pipefail
# -e  : exit immediately on command failure (non-zero rc)
# -u  : error on unset variables
# -o pipefail : pipeline fails if any command in pipeline fails

# =============================================================================
# CONFIGURATION / RUNTIME FLAGS
# =============================================================================
LOG_LEVEL_DEFAULT="INFO"                     # default logging verbosity
LOG_LEVEL="${LOG_LEVEL:-$LOG_LEVEL_DEFAULT}" # allow override: LOG_LEVEL=DEBUG bash gitter.sh
NONINTERACTIVE="${NONINTERACTIVE:-0}"        # if 1, skip "press any key" pauses
START_EPOCH_S="$(date +%s)"                  # used to compute total elapsed time

# =============================================================================
# LOGGING HELPERS
# =============================================================================
_ts() { date '+%Y-%m-%d %H:%M:%S'; }

_log() {
  # Writes a structured log line to STDERR.
  # Example: "2026-02-02 21:53:53 INFO    status_before"
  local level="$1"; shift
  printf '%s %-7s %s\n' "$(_ts)" "$level" "$*" >&2
}

debug() { [[ "${LOG_LEVEL}" == "DEBUG" ]] && _log "DEBUG" "$*"; }
info()  { _log "INFO"  "$*"; }
warn()  { _log "WARN"  "$*"; }
error() { _log "ERROR" "$*"; }

_die()  { error "$*"; exit 1; }

# =============================================================================
# PAUSE / STEP CONFIRMATION
# =============================================================================
_pause() {
  # Stops before important steps so you can review current state.
  # Set NONINTERACTIVE=1 to skip pauses.
  local label="${1:-continue}"
  [[ "${NONINTERACTIVE}" == "1" ]] && { info "pause_skipped step=${label} (NONINTERACTIVE=1)"; return 0; }
  info "pause step=${label} (press any key to continue)"
  IFS= read -r -n 1 -s _
}

# =============================================================================
# SAFE COMMAND RUNNER
# =============================================================================
_run() {
  # Runs a command, captures stdout+stderr to a temp file, logs timing,
  # and prints command output on failure (or in DEBUG).
  local start_ns end_ns elapsed_ms rc out_file
  start_ns="$(date +%s%N)"
  out_file="$(mktemp -t gitter_cmd.XXXXXX)"
  debug "exec: $*"

  set +e
  "$@" >"${out_file}" 2>&1
  rc="$?"
  set -e

  end_ns="$(date +%s%N)"
  elapsed_ms="$(( (end_ns - start_ns) / 1000000 ))"

  if [[ "${rc}" -ne 0 ]]; then
    error "cmd_failed rc=${rc} elapsed_ms=${elapsed_ms} cmd=$*"
    if [[ "${LOG_LEVEL}" == "DEBUG" ]]; then
      sed -n '1,240p' "${out_file}" >&2
    else
      tail -n 60 "${out_file}" >&2
    fi
    rm -f "${out_file}"
    return "${rc}"
  fi

  info "cmd_ok elapsed_ms=${elapsed_ms} cmd=$*"
  [[ "${LOG_LEVEL}" == "DEBUG" ]] && sed -n '1,120p' "${out_file}" >&2
  rm -f "${out_file}"
  return 0
}

# =============================================================================
# GLOBAL EXIT LOG (always prints final rc and elapsed time)
# =============================================================================
_on_exit() {
  local rc="$?"
  local end_epoch_s elapsed_s
  end_epoch_s="$(date +%s)"
  elapsed_s="$(( end_epoch_s - START_EPOCH_S ))"
  [[ "${rc}" -eq 0 ]] && info "done rc=0 elapsed_s=${elapsed_s}" || error "done rc=${rc} elapsed_s=${elapsed_s}"
}
trap _on_exit EXIT

# =============================================================================
# PRE-FLIGHT SAFETY CHECKS
# =============================================================================
_require_git_repo() {
  # Confirms we are inside a git working tree.
  _pause "########################## verify_repo"
  _run git rev-parse --is-inside-work-tree >/dev/null || _die "not a git repo"
}

_require_branch_attached() {
  # Confirms HEAD is attached to a branch (not detached HEAD).
  _pause "##########################verify_branch"
  local branch
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  [[ -n "$branch" ]] || _die "cannot read current branch"
  [[ "$branch" != "HEAD" ]] || _die "detached HEAD (create/switch to a branch first)"
  info "current_branch=${branch}"
}

_require_upstream() {
  # Confirms current branch tracks a remote branch (needed for pull/push).
  _pause "##########################verify_upstream"
  local upstream
  upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  [[ -n "$upstream" ]] || _die "no upstream set (git push -u origin <branch>)"
  info "upstream_ref=${upstream}"
}

# =============================================================================
# REBASE / CONFLICT UTILITIES
# =============================================================================
_rebase_in_progress() {
  # Detects rebase state by checking git internals.
  local git_dir
  git_dir="$(git rev-parse --git-dir)"
  [[ -d "${git_dir}/rebase-merge" || -d "${git_dir}/rebase-apply" ]]
}

_conflicted_files() {
  # Lists files currently marked as conflicted (U).
  git diff --name-only --diff-filter=U 2>/dev/null || true
}

_force_remote_wins_and_continue_rebase() {
  # When rebase produces conflicts:
  #   - "remote wins" policy => checkout --theirs for conflicted files
  #   - stage everything
  #   - continue rebase
  _pause "##########################resolve_conflicts_remote_wins"
  while _rebase_in_progress; do
    mapfile -t files < <(_conflicted_files)
    (( ${#files[@]} > 0 )) || _die "rebase in progress but no conflicted files detected"

    warn "conflicts_detected count=${#files[@]} files=${files[*]}"
    _run git checkout --theirs -- "${files[@]}"
    _run git add -A

    if git rebase --continue >/dev/null 2>&1; then
      info "rebase_continue_ok"
      continue
    fi

    _rebase_in_progress || _die "rebase --continue failed (manual intervention required)"
  done
}

# =============================================================================
# REMOTE-FIRST SYNC (FETCH + PULL REBASE AUTOSTASH)
# =============================================================================
_pull_rebase_autostash_remote_wins() {
  # What this does:
  #   1) Fetch remote updates (download, no working-tree changes)
  #   2) Pull with rebase + autostash:
  #       - remote commits are applied first
  #       - your local uncommitted edits are auto-stashed and re-applied after
  #       - "-X theirs" forces "remote wins" in merge strategy during rebase
  #   3) If pull fails with conflicts and rebase started, resolve by remote-wins.
  _pause "##########################fetch_origin"

  local fetch_rc pull_rc
  set +e
  git fetch origin >/tmp/gitter_fetch.log 2>&1
  fetch_rc="$?"
  set -e

  if [[ "${fetch_rc}" -ne 0 ]]; then
    error "fetch_failed rc=${fetch_rc}; last_output:"
    tail -n 120 /tmp/gitter_fetch.log >&2
    exit "${fetch_rc}"
  fi
  info "fetch_ok"

  _pause "##########################pull_rebase_autostash"
  info "pull_rebase_autostash start"

  set +e
  git pull --rebase --autostash -X theirs >/tmp/gitter_pull.log 2>&1
  pull_rc="$?"
  set -e

  if [[ "${pull_rc}" -eq 0 ]]; then
    info "pull_rebase_autostash ok"
    [[ "${LOG_LEVEL}" == "DEBUG" ]] && sed -n '1,160p' /tmp/gitter_pull.log >&2
    return 0
  fi

  warn "pull_failed rc=${pull_rc}; last_output:"
  tail -n 120 /tmp/gitter_pull.log >&2

  if _rebase_in_progress; then
    warn "rebase_started=true; attempting remote-wins conflict resolution"
    _force_remote_wins_and_continue_rebase
    info "pull_rebase_autostash resolved_with_remote_wins"
    return 0
  fi

  _die "pull failed before starting rebase (manual intervention required)"
}

# =============================================================================
# MAIN: ORCHESTRATION
# =============================================================================
main() {
  # 1) Validate environment and tracking state
  _require_git_repo
  _require_branch_attached
  _require_upstream

  # 2) Show current local modifications BEFORE touching remote
  _pause "##########################status_before"
  info "status_before"
  git status -sb >&2 || true

  # 3) Update local branch from remote (remote-first), reapply local edits
  _pull_rebase_autostash_remote_wins

  # 4) Show status AFTER pulling remote + reapplying local edits
  _pause "##########################status_after"
  info "status_after"
  git status -sb >&2 || true

  # 5) OPTIONAL: commit and push local changes (only if there are changes)
  #    Note: this is the only section that "uploads" anything to remote.
  _pause "##########################commit_local_changes"
  if ! git diff --quiet || ! git diff --cached --quiet; then
    info "commit_needed=true; staging_all"
    git add -A
    info "committing"
    git commit -m "wip: apply local changes after remote update"
    _pause "##########################push_to_remote"
    info "pushing"
    git push
  else
    info "no_local_changes_to_commit"
  fi
}

main "$@"

