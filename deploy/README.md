# Tabula deployment

## Stack

```
browser --HTTPS--> Cloudflare edge --tunnel--> cloudflared --> Caddy :8080 --> uvicorn 127.0.0.1:8000
```

- **cloudflared** maintains a persistent QUIC tunnel to Cloudflare.
- **Caddy** terminates on loopback, enforces body limits, CSP, cache headers, trusted proxies.
- **uvicorn** runs the FastAPI app.

## Named tunnel (permanent, `tabula.yuxuanz.xyz`)

### Prerequisites

1. Cloudflare account with `danielyx.net` zone.
2. Tunnel created in Zero Trust → Networks → Tunnels (or `cloudflared tunnel create tabula`).
3. Tunnel credentials file placed at `~/.cloudflared/5ea4e618-321b-4567-be84-5a107a4650a3.json`.
4. `~/.cloudflared/config.yml` with ingress rules.
5. **DNS record** in the `danielyx.net` zone:
   - Type: **CNAME**
   - Name: **tabula**
   - Target: **5ea4e618-321b-4567-be84-5a107a4650a3.cfargotunnel.com**
   - Proxy status: **Proxied** (orange cloud)

### Start the stack

```bash
# 1. Start the app
.venv/bin/python -m webapp.server --host 127.0.0.1 --port 8000

# 2. In another shell, start the tunnel stack
deploy/public.sh start
```

### Stop / status

```bash
deploy/public.sh stop
deploy/public.sh status
```

### systemd (optional)

```bash
sudo cp deploy/caddy.service deploy/cloudflared.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now caddy cloudflared
```

## Quick tunnel (ephemeral, `*.trycloudflare.com`)

If you don't have the DNS set up yet, use the old quick tunnel:

```bash
deploy/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8080
```

It prints a random `https://<words>.trycloudflare.com` URL. The old `cloudflared-quick.service` and the quick-tunnel logic in `public.sh` are kept for reference.

## Files

| File | Purpose |
|------|---------|
| `Caddyfile` | Reverse proxy config (binds 127.0.0.1:8080, trusts loopback proxy) |
| `cloudflared.service` | systemd unit for named tunnel |
| `cloudflared-quick.service` | systemd unit for quick tunnel (legacy) |
| `public.sh` | no-systemd start/stop/status for named tunnel |
| `install.sh` | fetches Caddy + cloudflared binaries into `deploy/bin/` |
| `tabula.service` | systemd unit for the uvicorn app |
| `bin/` | git-ignored Caddy / cloudflared binaries |
| `logs/` | git-ignored runtime logs |

## App environment

```bash
TABULA_HOST=127.0.0.1
TABULA_PUBLIC=1
TABULA_ALLOW_URLS=0
TABULA_MAX_AUDIO_MB=20
```

The app reads the real client IP from `X-Forwarded-For` (trusted because Caddy is the only upstream and `trusted_proxies static 127.0.0.1/32` is set). Per-IP budgets work correctly through the tunnel.
