#!/bin/bash
# Restart-on-exit supervisor for main.py.
#
# Real incident (2026-08-30): this bot has no process supervisor - when it
# hangs or crashes, it stays down until a human notices and restarts it by
# hand (confirmed: 3 separate hangs in ~24h, one lasting 3.2 hours, every
# open position unmonitored the whole time). Combined with the
# BINANCE_REQUEST_TIMEOUT_SECONDS fix (config.py/exchange.py - turns an
# indefinite REST hang into a bounded exception), this script is the other
# half: whatever the process does (clean exit, uncaught exception, killed),
# it gets relaunched automatically instead of silently staying down.
#
# Usage: run this INSTEAD of `venv/bin/python main.py` directly, e.g. from
# inside the same screen session:
#   ./run_bot.sh
cd "$(dirname "$0")" || exit 1

SUPERVISOR_LOG="logs/supervisor.log"
mkdir -p logs

while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - starting main.py" >> "$SUPERVISOR_LOG"

    venv/bin/python main.py
    exit_code=$?

    echo "$(date '+%Y-%m-%d %H:%M:%S') - main.py exited (code $exit_code) - restarting in 10s" >> "$SUPERVISOR_LOG"
    sleep 10
done
