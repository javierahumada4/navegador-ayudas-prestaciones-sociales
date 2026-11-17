#!/usr/bin/env bash
# Unattended job queue for the DGX (no SSH there, so work arrives through GitHub).
#
# Every INTERVAL seconds: pull BRANCH into $ARCA_WORK_DIR/arca-runner, run each jobs/*.sh with
# no log yet (name order, one at a time, env OUT=jobs/done/<name>), commit the log plus
# whatever the job left in $OUT and push it back. Start it once per DGX session, in tmux:
#
#     export ARCA_WORK_DIR=$HOME; git clone -q https://github.com/javierahumada4/navegador-ayudas-prestaciones-sociales.git ~/arca-runner 2>/dev/null; cd ~/arca-runner && git fetch -q origin feat/night-improvements && git checkout -q -B feat/night-improvements origin/feat/night-improvements && bash smoke/dgx_runner.sh feat/night-improvements
#
# The runner runs from a private copy and re-executes itself when the branch changes this
# file, so it can be fixed remotely. Anyone who can push to BRANCH runs code here: keep it a
# team branch and stop it when idle (Ctrl+C, or pkill -f dgx_runner).
set -u
[ -n "${RUNNER_DEBUG:-}" ] && set -x  # RUNNER_DEBUG=1: trace every command (survives re-exec)
export GIT_TERMINAL_PROMPT=0  # fail instead of hanging on a credential prompt
BRANCH="${1:?usage: dgx_runner.sh <branch> [interval_seconds]}"
INTERVAL="${2:-120}"
if [ -z "${ARCA_WORK_DIR:-}" ]; then
    ARCA_WORK_DIR="$HOME/clusters/dgx"; [ -d "$ARCA_WORK_DIR" ] || ARCA_WORK_DIR="$HOME"
fi
export ARCA_WORK_DIR
CLONE="$ARCA_WORK_DIR/arca-runner"
SELF="$ARCA_WORK_DIR/.dgx_runner.sh"

# Run from a copy: git checkout rewrites the repo file under a running bash.
if [ "$0" != "$SELF" ]; then cp "$0" "$SELF"; exec bash "$SELF" "$@"; fi

[ -d "$CLONE/.git" ] || git clone -q https://github.com/javierahumada4/navegador-ayudas-prestaciones-sociales.git "$CLONE"
cd "$CLONE" || exit 1
source smoke/dgx_env.sh >/dev/null || true
git config user.name >/dev/null || git config user.name dgx-runner
git config user.email >/dev/null || git config user.email dgx-runner@users.noreply.github.com
if [ -f "$ARCA_WORK_DIR/.env" ]; then set -a; . "$ARCA_WORK_DIR/.env"; set +a; fi
[ -f .env ] && { set -a; . ./.env; set +a; }

if ! git push -q --dry-run origin "HEAD:refs/heads/$BRANCH" 2>/dev/null; then
    echo "!!! git push no funciona desde aquí: los resultados no llegarán a GitHub."
    echo "!!! Arréglalo una vez (docs/github.md: token o 'gh auth login') y relanza."
fi
echo "[$(date -Is)] runner up on $BRANCH, every ${INTERVAL}s"
# Low-memory installs: a parallel torch+CUDA unpack can trip the session's cgroup OOM killer.
export UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1
# Diagnostics first, pushed before any heavy job, so a killed session still leaves evidence.
mkdir -p jobs/done
{ date -Is; hostname; nproc; free -g; df -h "$HOME" | tail -1
  cat /sys/fs/cgroup/memory.max 2>/dev/null; ulimit -a | grep -E "memory|processes"
  nvidia-smi -L 2>/dev/null; command -v uv; } > jobs/done/runner.log 2>&1
git add -f jobs/done/runner.log && git commit -qm "update: dgx runner up" --no-verify \
    && { git pull -q --rebase origin "$BRANCH" && git push -q origin "HEAD:$BRANCH"; }

while true; do
    git fetch -q origin "$BRANCH" && git checkout -q -B "$BRANCH" "origin/$BRANCH"
    cmp -s smoke/dgx_runner.sh "$SELF" || { echo "runner updated, restarting"; cp smoke/dgx_runner.sh "$SELF"; exec bash "$SELF" "$@"; }
    for job in jobs/*.sh; do
        [ -e "$job" ] || continue
        name="$(basename "$job" .sh)"
        log="jobs/done/$name.log"
        [ -e "$log" ] && continue
        mkdir -p "jobs/done/$name"
        echo "[$(date -Is)] start $name"
        { echo "commit $(git rev-parse --short HEAD) start $(date -Is)"
          OUT="jobs/done/$name" timeout "${JOB_TIMEOUT:-50400}" bash "$job"
          echo "exit=$? end $(date -Is)"; } > "$log" 2>&1
        echo "[$(date -Is)] done $name: $(tail -1 "$log")"
        git add -f "$log" "jobs/done/$name"
        git commit -qm "update: dgx job $name" --no-verify
        until git pull -q --rebase origin "$BRANCH" && git push -q origin "HEAD:$BRANCH"; do sleep 30; done
        continue 2  # re-fetch between jobs: newer pushes may fix the queue
    done
    sleep "$INTERVAL"
done
