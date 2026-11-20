# Public deployment

This directory contains everything needed to expose the Tabula workbench to the
internet from a machine with no domain name:

```
browser --HTTPS--> Cloudflare edge --quick tunnel--> cloudflared
                                                         |
                                                         v
                                              Caddy 127.0.0.1:8080
                                                         |
                                                         v
                                              uvicorn 127.0.0.1:8000
```

Caddy is mandatory in the middle: it caps request bodies, compresses the 1.2 MB
alphaTab bundle, sets the CSP and cache headers, and keeps Cloudflare pointed at
one stable local port. Never point `cloudflared` straight at uvicorn.

Files:

| File | Purpose |
| --- | --- |
| `Caddyfile` | proxy, compression, headers, cache policy, access log |
| `tabula.service` | uvicorn on `127.0.0.1:8000` |
| `caddy.service` | Caddy reading `/etc/caddy/Caddyfile` |
| `cloudflared-quick.service` | anonymous quick tunnel to Caddy |
| `install.sh` | downloads Caddy + cloudflared into `deploy/bin/` (no sudo) |
| `public.sh` | run/stop the whole stack without systemd |

## 1. Prerequisites

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r webapp/requirements-web.txt
.venv/bin/python -c "import fastapi, uvicorn, torch"
```

The app also needs FluidSynth, a soundfont and the fine-tuned checkpoint
(`runs/finetune_t5_guitarset/best`, overridable with `TABULA_CHECKPOINT`).
Start it once by hand to confirm the real error messages:

```bash
.venv/bin/python3 -m webapp.server --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/api/health
```

`python -m webapp.server` uses the repo root as cwd. `TABULA_ALLOW_URLS=0`
disables link downloads, which is the public default.

## 2. Install the binaries

```bash
deploy/install.sh          # idempotent; no sudo, starts nothing
deploy/install.sh --force  # re-download after a Caddy/cloudflared update
```

It fetches `caddy` and `cloudflared` (linux-amd64) into `deploy/bin/`, creates
`deploy/logs/`, `webapp/jobs/` and `data/cache/`, and prints the remaining
steps. If `/var/lib/tabula/jobs` cannot be created without root it prints the
one-time `sudo install -d -o "$USER" ...` command instead of running sudo.

Alternative if you prefer distro packages:

```bash
sudo apt install -y caddy
# then change ExecStart in caddy.service from
#   /home/daniel/Projects/tabula/deploy/bin/caddy
# to
#   /usr/bin/caddy
```

`cloudflared` is also packaged by Cloudflare, but `deploy/install.sh` is the
simplest path and pins nothing except "latest".

## 3. Quick tunnel (no domain, no account)

Terminal 1 — the app, with the public defaults the systemd unit would set:

```bash
cd /home/daniel/Projects/tabula
TABULA_PUBLIC=1 TABULA_ALLOW_URLS=0 TABULA_MAX_AUDIO_MB=20 \
  .venv/bin/python3 -m webapp.server --host 127.0.0.1 --port 8000
```

Terminal 2 — Caddy and the tunnel:

```bash
cd /home/daniel/Projects/tabula
deploy/public.sh
```

`public.sh` checks `/api/health`, starts Caddy on `127.0.0.1:8080`, starts
`cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8080`, waits for the
URL, and prints something like:

```
Public URL: https://random-words-here.trycloudflare.com
```

The URL is also in `deploy/logs/cloudflared.log` and via `deploy/public.sh
status`. It is random and changes on every start. `deploy/public.sh stop` kills
both processes using `deploy/logs/*.pid`; `deploy/public.sh restart` does both.

### With systemd instead

```bash
sudo install -d /etc/caddy
sudo ln -sfn /home/daniel/Projects/tabula/deploy/Caddyfile /etc/caddy/Caddyfile
sudo cp deploy/tabula.service deploy/caddy.service deploy/cloudflared-quick.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tabula caddy cloudflared-quick
```

The symlink keeps the unit's `/etc/caddy/Caddyfile` path while edits in the repo
take effect after `sudo systemctl reload caddy`. The Caddy unit runs as
`daniel` and stores state in `/var/lib/tabula-caddy`; `tabula.service` also
declares `TABULA_JOBS_DIR=/var/lib/tabula/jobs`, so create it once:

```bash
sudo install -d -o daniel -g daniel /var/lib/tabula/jobs
```

If it is missing the app cannot create `/var/lib` itself and the service will
restart-loop. `install.sh` prints this command when it cannot create the
directory without root.

For the service tunnel, filter the journal down to just the URL:

```bash
journalctl -u cloudflared-quick -f | grep --line-buffered -Eo 'https://[a-z0-9-]+\.trycloudflare\.com'
```

Because a quick tunnel has no account, it gets **no** WAF, no bot protection and
no Cloudflare Access. Treat the URL as public. It also has no uptime promise.

### Keep the tailnet/LAN listener

Caddy binds only loopback by default. To also serve the Tailscale address,
extend the `bind` line in the site block:

```caddyfile
    bind 127.0.0.1 100.111.69.56
```

The site block stays host-agnostic (`:8080`), so both the tunnel hostname and
the tailnet address match without extra site blocks. Caddy fails to start if any
listed address is absent, so only add `100.111.69.56` when `tailscale ip -4`
reports it. The tunnel still points at `http://127.0.0.1:8080`; the extra
listener only affects direct tailnet access. `http://100.111.69.56:8080` is
plain HTTP — use it on the tailnet, not over the open internet.

## 4. Restart, stop, verify

| Task | Quick tunnel | systemd |
| --- | --- | --- |
| start | `deploy/public.sh` | `sudo systemctl start tabula caddy cloudflared-quick` |
| status | `deploy/public.sh status` | `systemctl status tabula caddy cloudflared-quick` |
| restart | `deploy/public.sh restart` | `sudo systemctl restart tabula caddy cloudflared-quick` |
| stop | `deploy/public.sh stop` | `sudo systemctl stop cloudflared-quick caddy tabula` |
| public URL | `deploy/logs/cloudflared.log` | `journalctl -u cloudflared-quick -n 40` |

Validate a Caddyfile edit before reloading:

```bash
deploy/bin/caddy validate --config deploy/Caddyfile --adapter caddyfile
# systemd path:
deploy/bin/caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
```

After an app update (`git pull`, checkpoint swap), restart `tabula`; Caddy does
not need a restart. HTML and `app.js` are served with
`Cache-Control: max-age=0, must-revalidate`, `style.css`/`label.css` likewise,
while other `/static/*` assets are cached for a week and immutable. Bump the
file name (or wait out the week) if you change a cached asset in place.

Log locations:

- quick tunnel: `deploy/logs/caddy.log`, `deploy/logs/cloudflared.log`
- systemd: `journalctl -u tabula -f`, `-u caddy`, `-u cloudflared-quick`
- app jobs: `${TABULA_JOBS_DIR:-webapp/jobs}/<32-hex-id>/out/status.json` (poll
  this) and `out/report.json`

## 5. The CSP / Web Worker gotcha

alphaTab renders in a module Web Worker and uses an AudioWorklet for playback,
both loaded from the same origin, and in some code paths wraps the worker in a
`blob:` URL. The CSP in `Caddyfile` therefore keeps:

```
script-src 'self'
worker-src 'self' blob:
child-src  'self' blob:
```

If you tighten or extend the CSP:

- do not add a CDN script without also allowing it in `script-src`; alphaTab
  imports its own modules from `/static/alphatab/`;
- keep `worker-src` **and** `child-src` (Safari falls back to `child-src`);
- `connect-src 'self'` covers fetching `sonivox.sf2` and job files; an external
  sample host would need to be added there;
- `media-src 'self' blob:` is required because the player creates blob URLs for
  audio;
- `style-src 'unsafe-inline'` is required by alphaTab's inline styles.

A missing `worker-src`/`child-src` or a stray `default-src` restriction produces
a blank score and console errors about the worker, not an HTTP error. Test the
score and playback after any proxy or Cloudflare setting change. Cloudflare
features that inject scripts (Rocket Loader, some bot-fight challenges) will
also violate `script-src 'self'`; leave Rocket Loader off.

## 6. Abuse controls

A quick tunnel is an open door: anyone with the URL can upload audio and burn
CPU. Cloudflare's free edge gives you TLS and DDoS absorption but not a WAF for
`trycloudflare.com`. The controls therefore live in the app:

- **Upload cap** — `TABULA_MAX_AUDIO_MB=20` (unit file); the proxy allows 25 MB
  of multipart body so the app returns its own readable error. Upload size is
  checked before the file is written.
- **Conversion queue** — a single worker thread converts one take at a time and
  at most two more wait (`webapp/job_queue.py`). A burst beyond that gets a
  clear 429/queue-full response instead of N concurrent GPU jobs. `/api/convert`
  returns 202 immediately; the browser polls `out/status.json`, so no long-lived
  response ties up a worker or a Cloudflare connection.
- **Per-IP budget** — SQLite-backed counts in `webapp/job_queue.py`, default
  6 conversions per hour and 20 per day per address
  (`TABULA_RATE_PER_HOUR`, `TABULA_RATE_PER_DAY`; `0` disables a window). The
  address comes from `X-Forwarded-For`, which is why Caddy sets
  `trusted_proxies static 127.0.0.1/32` — without it every visitor would look
  like the loopback proxy and share one budget.
- **Job isolation** — job ids are 32 hex characters, job file names are an
  allow-list, and jobs are removed after `TABULA_JOB_RETENTION_DAYS` (default
  7), with a total cap of `TABULA_JOBS_MAX_GB` (default 5).
- **No remote fetches** — `TABULA_ALLOW_URLS=0` removes the "download from a
  link" path entirely.
- **Quiet API** — `TABULA_PUBLIC=1` disables `/api/docs` and the OpenAPI schema
  so the public hostname does not advertise the API surface.

The service unit sets `TABULA_PUBLIC=1`, so these public defaults apply to both
the systemd and quick-tunnel paths (the quick-tunnel script reuses the same app
process). For a link you intend to share for more than a demo, prefer a named
tunnel with Cloudflare Access (section 7) over relying on a random URL as a
secret.

## 7. Moving to a named tunnel when a domain exists

Quick tunnels are for demos. With a domain on Cloudflare:

```bash
deploy/bin/cloudflared tunnel login
deploy/bin/cloudflared tunnel create tabula
deploy/bin/cloudflared tunnel route dns tabula tab.example.com
```

`~/.cloudflared/config.yml`:

```yaml
tunnel: <tunnel-uuid>
credentials-file: /home/daniel/.cloudflared/<tunnel-uuid>.json
ingress:
  - hostname: tab.example.com
    service: http://127.0.0.1:8080
  - service: http_status:404
```

Then either run `cloudflared --no-autoupdate tunnel run tabula` or edit
`cloudflared-quick.service`: remove `--url http://127.0.0.1:8080`, use
`ExecStart=.../cloudflared --no-autoupdate tunnel run tabula`, and allow home
reads (`ProtectHome=read-only` is fine; `ProtectHome=yes` would hide
`~/.cloudflared/`). Rename the unit or keep it; the tunnel no longer prints a
URL because the hostname is fixed. Add Cloudflare Access (Zero Trust →
Access → Applications) for authentication — the app itself has no login. If
Cloudflare proxies the hostname, disable caching for `/api/*` and keep Rocket
Loader off so the CSP stays valid.

## 8. Troubleshooting

| Symptom | Check |
| --- | --- |
| 502 from Caddy | uvicorn is down or not on `127.0.0.1:8000`; `curl localhost:8000/api/health` |
| Caddy will not start | `deploy/bin/caddy validate ...`; port 8080 already in use; tailnet address listed but not up |
| 404 from Caddy through the tunnel | site block must stay host-agnostic (`:8080` + `bind 127.0.0.1`); cloudflared preserves the `*.trycloudflare.com` Host header |
| Tunnel prints no URL | `deploy/logs/cloudflared.log`; outbound HTTPS/DNS blocked; try `deploy/install.sh --force` for a newer binary |
| URL loads but score is blank | CSP/worker issue; browser console, section 5 |
| Upload rejected at 413 | proxy cap 25 MB or app cap 20 MB; both intentional |
| `public.sh` says already running | stale pid file; `deploy/public.sh status`, then `stop` |
| Changes not visible | hard refresh; HTML/JS are revalidated, other static files are cached for a week |
| Permission denied writing jobs | service user/ownership of `webapp/jobs` or `TABULA_JOBS_DIR` |

Do not commit `deploy/bin/` or `deploy/logs/`; they are ignored by
`deploy/.gitignore`. No secrets are stored in this directory.
