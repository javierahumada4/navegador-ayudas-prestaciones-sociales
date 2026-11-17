#!/usr/bin/env bash
# Unattended job queue for the DGX (no SSH there, so work arrives through GitHub).
#
# Every INTERVAL seconds: pull BRANCH, run each jobs/*.sh with no log yet (name order, one at
# a time, env OUT=jobs/done/<name>), commit the log plus whatever the job left in $OUT and push
# it back. Start it from a clone of the repo, in the DGX terminal:
#
#     cd ~/arca-runner && bash smoke/dgx_runner.sh feat/night-improvements
#
# Every step is printed and appended to ~/runner-steps.log, so a dead session leaves a trace.
# Anyone who can push to BRANCH runs code here: keep it a team branch, stop it when idle.
# The whole script is one function: bash parses it fully before running, so a git checkout
# that rewrites this file mid-run cannot corrupt the running copy.

step() { echo "[runner $(date +%T)] $*" | tee -a "$HOME/runner-steps.log" >&2; }

main() {
    step "1 start: shell=$BASH_VERSION pwd=$PWD args=$*"
    BRANCH="${1:?usage: dgx_runner.sh <branch> [interval_seconds]}"
    INTERVAL="${2:-120}"
    export GIT_TERMINAL_PROMPT=0  # fail instead of hanging on a credential prompt
    if [ -z "${ARCA_WORK_DIR:-}" ]; then
        ARCA_WORK_DIR="$HOME/clusters/dgx"; [ -d "$ARCA_WORK_DIR" ] || ARCA_WORK_DIR="$HOME"
    fi
    export ARCA_WORK_DIR
    step "2 work dir $ARCA_WORK_DIR; repo $(git rev-parse --show-toplevel 2>&1)"
    cd "$(git rev-parse --show-toplevel)" || { step "not inside the repo clone"; return 1; }

    step "3 loading smoke/dgx_env.sh"
    source smoke/dgx_env.sh || step "dgx_env.sh returned $?"
    step "4 env loaded: uv=$(command -v uv) HF_HOME=${HF_HOME:-}"
    git config user.name >/dev/null || git config user.name dgx-runner
    git config user.email >/dev/null || git config user.email dgx-runner@users.noreply.github.com
    for f in "$ARCA_WORK_DIR/.env" ./.env; do
        [ -f "$f" ] && { step "5 sourcing $f"; set -a; . "$f"; set +a; }
    done
    export UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1

    step "6 checking git push"
    if ! git push -q --dry-run origin "HEAD:refs/heads/$BRANCH"; then
        step "!!! git push no funciona: los resultados no llegarán a GitHub"
    fi
    step "7 host diagnostics"
    mkdir -p jobs/done
    { date -Is; hostname; nproc; free -g; df -h "$HOME" | tail -1
      cat /sys/fs/cgroup/memory.max 2>/dev/null; ulimit -a | grep -E "memory|processes"
      nvidia-smi -L 2>/dev/null; command -v uv; } > jobs/done/runner.log 2>&1
    git add -f jobs/done/runner.log && git commit -qm "update: dgx runner up" --no-verify
    push
    step "8 runner up on $BRANCH, every ${INTERVAL}s"

    local self_hash
    self_hash="$(git hash-object smoke/dgx_runner.sh)"
    while true; do
        # Rebase, never reset: job results whose push failed must survive until a later push.
        if git fetch -q origin "$BRANCH"; then
            git rebase -q "origin/$BRANCH" || { git rebase --abort; step "rebase failed"; }
            git log -1 --format=%s | grep -q "^update: dgx" && ! git diff --quiet "origin/$BRANCH" HEAD \
                && push
        fi
        if [ "$(git hash-object smoke/dgx_runner.sh)" != "$self_hash" ]; then
            step "runner updated on the branch, restarting"
            exec bash smoke/dgx_runner.sh "$@"
        fi
        for job in jobs/*.sh; do
            [ -e "$job" ] || continue
            name="$(basename "$job" .sh)"
            log="jobs/done/$name.log"
            [ -e "$log" ] && continue
            mkdir -p "jobs/done/$name"
            step "job $name: start"
            { echo "commit $(git rev-parse --short HEAD) start $(date -Is)"
              OUT="jobs/done/$name" timeout "${JOB_TIMEOUT:-50400}" bash "$job"
              echo "exit=$? end $(date -Is)"; } > "$log" 2>&1
            step "job $name: $(tail -1 "$log")"
            git add -f "$log" "jobs/done/$name"
            git commit -qm "update: dgx job $name" --no-verify
            push
            continue 2  # re-fetch between jobs: newer pushes may fix the queue
        done
        sleep "$INTERVAL"
    done
}

push() {
    local tries=0
    until git pull -q --rebase origin "$BRANCH" && git push -q origin "HEAD:$BRANCH"; do
        tries=$((tries + 1)); step "push failed ($tries), retrying in 30 s"
        [ "$tries" -ge 20 ] && { step "giving up on this push"; return 1; }
        sleep 30
    done
    step "pushed $(git rev-parse --short HEAD)"
}

main "$@"
exit
