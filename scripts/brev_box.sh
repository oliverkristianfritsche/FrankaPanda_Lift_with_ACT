#!/usr/bin/env bash
# Oliver Fritsche — June 7, 2026 — CS 7180 Advanced Perception
#
# Park / resume the Isaac Brev launchable cheaply and safely, to avoid the
# expensive delete->redeploy rebuild (~15 min) and stuck stops.
#
#   scripts/brev_box.sh park      # kill Isaac, then stop the box (halts GPU billing)
#   scripts/brev_box.sh resume    # start the box, wait READY, ensure containers + pull repo
#   scripts/brev_box.sh status    # show instance + container + isaac-process state
#
# Why stop (not delete): stopping preserves the built containers, the cloned
# repo, the pip install, the PPO checkpoint, AND the shader cache, so `resume`
# is ~1-3 min with NO rebuild and a faster Isaac boot. We kill the Isaac process
# FIRST so the stop doesn't hang the way it did when a live render job + an
# unhealthy box blocked Brev's graceful teardown.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

CMD="${1:-status}"
INSTANCE="${2:-$(brev ls --json 2>/dev/null | grep -oE 'isaac-launchable-[a-zA-Z0-9]+' | head -1)}"
[ -z "${INSTANCE:-}" ] && { echo "No isaac-launchable instance found (pass the name as the 2nd arg)."; exit 1; }
COMPOSE_DIR="~/isaac-launchable/isaac-lab"

case "$CMD" in
  park)
    echo ">> Quiescing Isaac on $INSTANCE (free the GPU so the stop doesn't hang)..."
    brev exec "$INSTANCE" "docker exec vscode bash -lc \"pkill -9 -f 'kit/python' 2>/dev/null; pkill -9 -f python.sh 2>/dev/null; true\"" 2>/dev/null || echo "   (quiesce best-effort; continuing)"
    echo ">> brev stop $INSTANCE"
    brev stop "$INSTANCE"
    echo ">> Stopping. Run 'scripts/brev_box.sh status' in ~1 min; GPU billing ends once STOPPED."
    ;;
  resume)
    echo ">> brev start $INSTANCE"
    brev start "$INSTANCE" 2>&1 | tail -2
    echo ">> Waiting for RUNNING + COMPLETED + READY..."
    for i in $(seq 1 30); do
      L=$(brev ls 2>/dev/null | grep "$INSTANCE")
      echo "   $(date +%H:%M:%S) $(echo "$L" | awk '{print $2, $3, $4}')"
      if echo "$L" | grep -q RUNNING && echo "$L" | grep -q READY; then break; fi
      sleep 20
    done
    echo ">> Ensuring containers are up and pulling latest repo..."
    brev exec "$INSTANCE" "cd $COMPOSE_DIR && docker compose up -d 2>&1 | tail -4; docker exec vscode bash -lc 'cd /workspace/repo && git pull 2>&1 | tail -2'" 2>&1 | grep -vE 'waiting for SSH' | tail -10
    echo ">> Ready. Containers (incl. the /viewer stack) are back up; resume your work."
    ;;
  status)
    brev ls 2>&1 | grep -E "NAME|$INSTANCE"
    echo "--- containers / isaac procs ---"
    brev exec "$INSTANCE" "docker ps --format '{{.Names}} | {{.Status}}' 2>/dev/null; echo -n 'isaac processes: '; docker exec vscode bash -lc 'ps aux | grep -E \"kit/python|python.sh\" | grep -v grep | wc -l' 2>/dev/null" 2>/dev/null | grep -vE 'waiting for SSH'
    ;;
  *)
    echo "usage: $0 {park|resume|status} [instance-name]"; exit 1 ;;
esac
