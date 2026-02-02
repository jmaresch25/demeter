#!/usr/bin/env bash
set -euo pipefail

LOG_LEVEL_DEFAULT="INFO"
LOG_LEVEL="${LOG_LEVEL:-$LOG_LEVEL_DEFAULT}"
NONINTERACTIVE="${NONINTERACTIVE:-0}"
START_EPOCH_S="$(date +%s)"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
_log() { local level="$1"; shift; printf '%s %-7s %s\n' "$(_ts)" "$level" "$*" >&2; }
debug() { [[ "${LOG_LEVEL}" == "DEBUG" ]] && _log "DEBUG" "$*"; }
info()  { _log "INFO"  "$*"; }
warn()  { _log "WARN"  "$*"; }
error() { _log "ERROR" "$*"; }
_die()  { error "$*"; exit 1; }

_pause() {
  local label="${1:-continue}"
  [[ "${NONINTERACTIVE}" == "1" ]] && { info "pause_skipped step=${label} (NONINTERACTIVE=1)"; return 0; }
  info "pause step=${label} (press any key to continue)"
  IFS= read -r -n 1 -s _
}

_run() {
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

_on_exit() {
  local rc="$?"
  local end_epoch_s elapsed_s
  end_epoch_s="$(date +%s)"
  elapsed_s="$(( end_epoch_s - START_EPOCH_S ))"
  [[ "${rc}" -eq 0 ]] && info "done rc=0 elapsed_s=${elapsed_s}" || error "done rc=${rc} elapsed_s=${elapsed_s}"
}
trap _on_exit EXIT

_require_git_repo() {
  _pause "verify_repo"
  _run git rev-parse --is-inside-work-tree >/dev/null || _die "not a git repo"
}

_require_branch_attached() {
  _pause "verify_branch"
  local branch
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  [[ -n "$branch" ]] || _die "cannot read current branch"
  [[ "$branch" != "HEAD" ]] || _die "detached HEAD (create/switch to a branch first)"
  info "current_branch=${branch}"
}

_require_upstream() {
  _pause "verify_upstream"
  local upstream
  upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  [[ -n "$upstream" ]] || _die "no upstream set (git push -u origin <branch>)"
  info "upstream_ref=${upstream}"
}

_rebase_in_progress() {
  local git_dir
  git_dir="$(git rev-parse --git-dir)"
  [[ -d "${git_dir}/rebase-merge" || -d "${git_dir}/rebase-apply" ]]
}

_conflicted_files() { git diff --name-only --diff-filter=U 2>/dev/null || true; }

_force_remote_wins_and_continue_rebase() {
  _pause "resolve_conflicts_remote_wins"
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


_pull_rebase_autostash_remote_wins() {
  _pause "fetch_origin"

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

  _pause "pull_rebase_autostash"
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

_pull_rebase_autostash_remote_wins() {
  _pause "fetch_origin"

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

  _pause "pull_rebase_autostash"
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




main() {
  _require_git_repo
  _require_branch_attached
  _require_upstream

  _pause "status_before"
  info "status_before"
  git status -sb >&2 || true

  _pull_rebase_autostash_remote_wins

  _pause "status_after"
  info "status_after"
  git status -sb >&2 || true

  _pause "commit_local_changes"
  if ! git diff --quiet || ! git diff --cached --quiet; then
    git add -A
    git commit -m "wip: apply local changes after remote update"
    _pause "push_to_remote"
    git push
  else
    info "no_local_changes_to_commit"
  fi
}

main "$@"
