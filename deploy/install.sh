#!/usr/bin/env bash
# Install the two public-deployment binaries for Tabula into deploy/bin/ and
# prepare the directories the services expect.
#
# This script deliberately does NOT use sudo, does NOT install systemd units,
# and does NOT start any process.  It is safe to re-run: existing binaries are
# left alone unless --force is passed.
#
# Usage:
#   deploy/install.sh            # download missing pieces, create dirs
#   deploy/install.sh --force    # re-download caddy and cloudflared
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN_DIR="${SCRIPT_DIR}/bin"
LOG_DIR="${SCRIPT_DIR}/logs"

CADDY_URL="https://caddyserver.com/api/download?os=linux&arch=amd64"
CLOUDFLARED_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"

FORCE=0
for arg in "$@"; do
	case "$arg" in
	--force) FORCE=1 ;;
	-h | --help)
		sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
		exit 0
		;;
	*)
		echo "Unknown option: $arg" >&2
		echo "Usage: deploy/install.sh [--force]" >&2
		exit 2
		;;
	esac
done

fetch() {
	# fetch <url> <destination>
	if command -v curl >/dev/null 2>&1; then
		curl -fL --retry 3 --connect-timeout 15 --max-time 300 -o "$2" "$1"
	elif command -v wget >/dev/null 2>&1; then
		wget -qO "$2" "$1"
	else
		echo "ERROR: need curl or wget on PATH." >&2
		return 1
	fi
}

install_binary() {
	# install_binary <name> <url> <version flag...>
	local name="$1" url="$2"
	shift 2
	local dest="${BIN_DIR}/${name}"
	local tmp="${dest}.download"

	if [ -x "$dest" ] && [ "$FORCE" -eq 0 ]; then
		echo "  ${name}: already present (${dest}), skipping"
		return 0
	fi

	echo "  ${name}: downloading ${url}"
	mkdir -p "$BIN_DIR"
	# Remove a stale partial download before retrying.
	rm -f "$tmp"
	fetch "$url" "$tmp"
	chmod 0755 "$tmp"
	if ! "$tmp" version >/dev/null 2>&1 && ! "$tmp" --version >/dev/null 2>&1; then
		rm -f "$tmp"
		echo "ERROR: ${name} downloaded but does not run; aborting." >&2
		return 1
	fi
	mv -f "$tmp" "$dest"
	echo "  ${name}: installed $(basename "$dest")"
}

echo "Tabula deployment installer"
echo "  repo:  ${REPO_DIR}"
echo "  bin:   ${BIN_DIR}"
echo

echo "Binaries"
install_binary caddy "$CADDY_URL"
install_binary cloudflared "$CLOUDFLARED_URL"
echo

echo "Directories"
mkdir -p "$LOG_DIR"
echo "  logs:  ${LOG_DIR}"
mkdir -p "${REPO_DIR}/webapp/jobs" "${REPO_DIR}/data/cache"
echo "  jobs:  ${REPO_DIR}/webapp/jobs (current server default)"
echo "  cache: ${REPO_DIR}/data/cache"

# tabula.service sets TABULA_JOBS_DIR=/var/lib/tabula/jobs.  Creating it needs
# root the first time; do not invoke sudo here, just report the one-liner.
JOBS_DIR="/var/lib/tabula/jobs"
if mkdir -p "$JOBS_DIR" 2>/dev/null; then
	echo "  jobs:  ${JOBS_DIR}"
else
	echo "  jobs:  ${JOBS_DIR} is not writable by $(id -un)."
	echo "         One-time, with sudo:"
	echo "           sudo install -d -o $(id -un) -g $(id -gn) ${JOBS_DIR}"
fi
echo

echo "Next steps"
echo "  1. Check the app starts on loopback:"
echo "       cd ${REPO_DIR} && TABULA_PUBLIC=1 TABULA_ALLOW_URLS=0 TABULA_MAX_AUDIO_MB=20 \\"
echo "         .venv/bin/python3 -m webapp.server --host 127.0.0.1 --port 8000"
echo "  2. For a public link without systemd:"
echo "       ${SCRIPT_DIR}/public.sh"
echo "       ${SCRIPT_DIR}/public.sh stop"
echo "  3. For systemd (needs sudo once; not done by this script):"
echo "       sudo install -d /etc/caddy"
echo "       sudo ln -sfn ${SCRIPT_DIR}/Caddyfile /etc/caddy/Caddyfile"
echo "       sudo cp ${SCRIPT_DIR}/{tabula,caddy,cloudflared-quick}.service /etc/systemd/system/"
echo "       sudo systemctl daemon-reload"
echo "       sudo systemctl enable --now tabula caddy cloudflared-quick"
echo "  4. Read ${SCRIPT_DIR}/README.md for the tunnel URL, logs and CSP notes."
