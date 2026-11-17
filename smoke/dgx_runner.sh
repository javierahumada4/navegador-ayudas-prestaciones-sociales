#!/usr/bin/env bash
# Unattended job queue for the DGX (no SSH there, so work arrives through GitHub).
#
# Every INTERVAL seconds: pull BRANCH into a private clone, run each jobs/*.sh that has no
# log yet (in name order, one at a time), then commit jobs/done/<name>.log plus whatever the
# job copied into jobs/done/<name>/ and push it back. Start it once per DGX session:
#
#     source smoke/dgx_env.sh
#     nohup bash smoke/dgx_runner.sh feat/night-improvements > "$HOME/arca-runner.out" 2>&1 &
#
# The clone lives in $ARCA_WORK_DIR/arca-runner, so it never touches the checkout that
# serves the API. Anyone who can push to BRANCH can run code here: keep it a team branch.
# Stop it with: pkill -f dgx_runner.sh
set -u
BRANCH="${1:?usage: dgx_runner.sh <branch> [interval_seconds]}"
INTERVAL="${2:-300}"
WORK_DIR="${ARCA_WORK_DIR:-$HOME/clusters/dgx}"
CLONE="$WORK_DIR/arca-runner"
REMOTE="$(git remote get-url origin)"

# Reuse the launching checkout's .env (HF token, adapter ids) for every job.
if [ -f .env ]; then set -a; . ./.env; set +a; fi
[ -d "$CLONE/.git" ] || git clone -q "$REMOTE" "$CLONE"
cd "$CLONE" || exit 1

while true; do
    git fetch -q origin "$BRANCH" && git checkout -q -B "$BRANCH" "origin/$BRANCH"
    for job in jobs/*.sh; do
        [ -e "$job" ] || continue
        name="$(basename "$job" .sh)"
        log="jobs/done/$name.log"
        [ -e "$log" ] && continue
        mkdir -p "jobs/done/$name"
        echo "[$(date -Is)] start $name" | tee -a "$HOME/arca-runner.out"
        { echo "commit $(git rev-parse --short HEAD) start $(date -Is)"
          OUT="jobs/done/$name" timeout "${JOB_TIMEOUT:-21600}" bash "$job"
          echo "exit=$? end $(date -Is)"; } > "$log" 2>&1
        git add -f "$log" "jobs/done/$name"
        git commit -qm "update: dgx job $name" --no-verify
        until git pull -q --rebase origin "$BRANCH" && git push -q origin "HEAD:$BRANCH"; do sleep 30; done
    done
    sleep "$INTERVAL"
done
