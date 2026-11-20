#!/usr/bin/env bash
# Start or stop the public quick-tunnel stack without systemd.
#
#   deploy/public.sh          start Caddy + cloudflared, print the public URL
#   deploy/public.sh stop     stop the processes started by this script
#   deploy/public.sh status   show pid/log state
#
# Caddy listens on 127.0.0.1:8080 and proxies to uvicorn on 127.0.0.1:8000.
# cloudflared creates an anonymous quick tunnel and prints a random
# https://*.trycloudflare.com URL.  The URL changes every start.
#
# This script does not start uvicorn; run the app first:
#   .venv/bin/python3 -m webapp.server --host 127.0.0.1 --port 8000
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN_DIR="${SCRIPT_DIR}/bin"
LOG_DIR="${SCRIPT_DIR}/logs"
CADDY="${BIN_DIR}/caddy"
CLOUDFLARED="${BIN_DIR}/cloudflared"
CADDY_CONFIG="${SCRIPT_DIR}/Caddyfile"
UPSTREAM="http://127.0.0.1:8000"
TUNNEL_TARGET="http://127.0.0.1:8080"

CADDY_PID="${LOG_DIR}/caddy.pid"
CLOUDFLARED_PID="${LOG_DIR}/cloudflared.pid"
CADDY_LOG="${LOG_DIR}/caddy.log"
CLOUDFLARED_LOG="${LOG_DIR}/cloudflared.log"

die() {
	echo "ERROR: $*" >&2
	exit 1
}

pid_alive() {
	local pidfile="$1" pid
	[ -f "$pidfile" ] || return 1
	pid="$(cat "$pidfile" 2>/dev/null || true)"
	[ -n "$pid" ] || return 1
	kill -0 "$pid" 2>/dev/null
}

stop_process() {
	# stop_process <name> <pidfile>
	local name="$1" pidfile="$2" pid i
	if ! pid_alive "$pidfile"; then
		rm -f "$pidfile"
		return 0
	fi
	pid="$(cat "$pidfile")"
	echo "  stopping ${name} (pid ${pid})"
	kill "$pid" 2>/dev/null || true
	for i in $(seq 1 50); do
		kill -0 "$pid" 2>/dev/null || break
		sleep 0.1
	done
	if kill -0 "$pid" 2>/dev/null; then
		echo "  ${name} did not exit; sending SIGKILL"
		kill -9 "$pid" 2>/dev/null || true
	fi
	rm -f "$pidfile"
}

stop() {
	echo "Stopping Tabula quick-tunnel stack"
	# Stop the tunnel first so visitors see a clean failure, then Caddy.
	stop_process cloudflared "$CLOUDFLARED_PID"
	stop_process caddy "$CADDY_PID"
	echo "Stopped."
}

status() {
	local name pidfile pid state
	echo "Tabula quick-tunnel status"
	for name in caddy cloudflared; do
		if [ "$name" = caddy ]; then pidfile="$CADDY_PID"; else pidfile="$CLOUDFLARED_PID"; fi
		if pid_alive "$pidfile"; then
			pid="$(cat "$pidfile")"
			state="running (pid ${pid})"
		else
			state="stopped"
		fi
		printf '  %-12s %s\n' "$name" "$state"
	done
	if [ -f "$CLOUDFLARED_LOG" ]; then
		local url
		url="$(grep -Eo 'https://[A-Za-z0-9.-]+\.trycloudflare\.com' "$CLOUDFLARED_LOG" | tail -n 1 || true)"
		[ -n "$url" ] && echo "  url          ${url}"
	fi
	echo "  logs         ${LOG_DIR}"
}

wait_http() {
	# wait_http <url> <seconds>
	local url="$1" seconds="$2" i
	for i in $(seq 1 "$seconds"); do
		if curl -fsS --max-time 2 -o /dev/null "$url"; then
			return 0
		fi
		sleep 1
	done
	return 1
}

start() {
	echo "Starting Tabula quick-tunnel stack"

	[ -x "$CADDY" ] || die "missing ${CADDY}; run deploy/install.sh first."
	[ -x "$CLOUDFLARED" ] || die "missing ${CLOUDFLARED}; run deploy/install.sh first."
	[ -f "$CADDY_CONFIG" ] || die "missing ${CADDY_CONFIG}."

	if pid_alive "$CADDY_PID" || pid_alive "$CLOUDFLARED_PID"; then
		die "already running; use 'deploy/public.sh stop' or 'status' first."
	fi

	command -v curl >/dev/null 2>&1 || die "curl is required."

	if ! curl -fsS --max-time 3 -o /dev/null "${UPSTREAM}/api/health"; then
		die "uvicorn is not answering on ${UPSTREAM}. Start it first:
  cd ${REPO_DIR} && .venv/bin/python3 -m webapp.server --host 127.0.0.1 --port 8000"
	fi

	mkdir -p "$LOG_DIR"
	: >"$CADDY_LOG"
	: >"$CLOUDFLARED_LOG"

	echo "  starting Caddy on 127.0.0.1:8080 (log: ${CADDY_LOG})"
	nohup "$CADDY" run --config "$CADDY_CONFIG" --adapter caddyfile >>"$CADDY_LOG" 2>&1 &
	echo "$!" >"$CADDY_PID"

	if ! wait_http "$TUNNEL_TARGET/" 15; then
		echo "--- ${CADDY_LOG} ---" >&2
		tail -n 40 "$CADDY_LOG" >&2 || true
		stop
		die "Caddy did not start. Is port 8080 already in use?"
	fi
	echo "  caddy ready (pid $(cat "$CADDY_PID"))"

	echo "  starting cloudflared (log: ${CLOUDFLARED_LOG})"
	nohup "$CLOUDFLARED" tunnel --no-autoupdate --url "$TUNNEL_TARGET" >>"$CLOUDFLARED_LOG" 2>&1 &
	echo "$!" >"$CLOUDFLARED_PID"

	echo "  waiting for the public URL (cloudflare side can take a few seconds)"
	url=""
	for i in $(seq 1 60); do
		url="$(grep -Eo 'https://[A-Za-z0-9.-]+\.trycloudflare\.com' "$CLOUDFLARED_LOG" | head -n 1 || true)"
		[ -n "$url" ] && break
		if ! pid_alive "$CLOUDFLARED_PID"; then
			break
		fi
		sleep 1
	done

	if [ -z "$url" ]; then
		echo "--- ${CLOUDFLARED_LOG} ---" >&2
		tail -n 40 "$CLOUDFLARED_LOG" >&2 || true
		stop
		die "cloudflared did not print a trycloudflare URL."
	fi

	echo
	echo "Public URL: ${url}"
	echo "  caddy:       pid $(cat "$CADDY_PID")  (${CADDY_PID})"
	echo "  cloudflared: pid $(cat "$CLOUDFLARED_PID")  (${CLOUDFLARED_PID})"
	echo "  logs:        ${LOG_DIR}"
	echo
	echo "Stop with: ${SCRIPT_DIR}/public.sh stop"
	echo "The URL is random and changes on every start."
}

case "${1:-start}" in
start | up) start ;;
stop | down) stop ;;
status) status ;;
restart)
	stop
	start
	;;
-h | --help | help)
	sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
	;;
*)
	echo "Usage: deploy/public.sh [start|stop|status|restart]" >&2
	exit 2
	;;
esac
