# Training on WATcloud

Cluster-specific notes for running Tabula on WATcloud, plus the upstream SLURM
documentation condensed to what this project needs.

Upstream: <https://cloud.watonomous.ca/docs/compute-cluster/slurm>
(last checked October 2026).

## Access

Login nodes are `wato-login1` and `wato-login2`:

```bash
ssh wato-login1.ext.watonomous.ca   # user: your WATcloud username
```

The Slurm CLI is **not** on the default login `PATH`:

```bash
export PATH=/opt/scripts/bin:$PATH
# or call binaries directly: /opt/scripts/bin/sbatch
```

If the cluster is only reachable from another machine on your network, put a
`ProxyJump` in `~/.ssh/config`:

```
Host watcloud
    HostName wato-login1.ext.watonomous.ca
    ProxyJump <your-jump-host>
    User <your-username>
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

Then `ssh watcloud`, `rsync`, and `scp` all work unchanged.

## Cluster layout

Two partitions, both spanning the same nodes:

| Partition | Max wall time | Use for |
| --- | --- | --- |
| `compute` (default) | 1 day | short/interactive work |
| `compute_dense` | 7 days | long training runs |

Check the current state yourself — it changes:

```bash
sinfo -o "%.20P %.10a %.12l %.6D %.10t %N"
scontrol show node | grep -E "NodeName|Gres=|Partitions=" | paste - - -
```

Observed nodes (October 2026):

| Node | GPUs | `shard` pool | Notes |
| --- | --- | --- | --- |
| `thor-slurm1` | 2 × RTX 3090 | 48K MiB | good default |
| `trpro-slurm1` | 4 × RTX 3090 | 96K MiB | good default |
| `trpro-slurm2` | 1 × RTX 4090 | 24K MiB | Ada, bf16 fine |
| `delta-slurm1` | 2 × RTX 2080 Ti | 22K MiB | Turing, bf16 fine |
| `tr-slurm2` | 1 × GTX 1080 | 8K MiB | **Pascal — no bf16** |
| `wato2-slurm1` | 1 × RTX 2080 Ti | 11K MiB | often drained |

**Always type the GPU in the GRES request.** An untyped `shard` request can
land on the GTX 1080 node, where bf16 silently falls back to fp32.

## GRES

`shard` is VRAM in MiB on a shared GPU; `gpu` is exclusive access.

```bash
# 8 GiB of an RTX 3090 (recommended)
srun --gres shard:rtx_3090:8192 --pty bash

# 8 GiB of any GPU
srun --gres shard:8192 --pty bash

# whole RTX 3090 — avoid unless the job really uses the card
srun --gres gpu:rtx_3090:1 --pty bash

# combine with temporary disk (mounted at /tmp, cleaned up at job end)
srun --gres shard:rtx_3090:8192,tmpdisk:20480 --pty bash
```

The requested `shard` size is **not enforced**. It is a scheduling hint, so
keep it honest — a 2.8M-parameter model does not need more than 8 GiB.

`tmpdisk` is a separate GRES, requested in MiB and mounted at `/tmp`. The job
script asks for 20 GiB.

## Submitting a training run

From the repository root on the login node:

```bash
export PATH=/opt/scripts/bin:$PATH
mkdir -p logs
sbatch scripts/train_t5.sh
```

Any variable the script reads can be overridden at submit time:

```bash
BATCH_SIZE=16 GRAD_ACCUM=4 MAX_STEPS=75000 sbatch scripts/train_t5.sh
```

`scripts/train_t5.sh` requests:

| Resource | Value | Why |
| --- | --- | --- |
| Partition | `compute_dense` | long jobs; 7-day max |
| Time | `2-00:00:00` | generous; the job exits at `--max-steps` |
| GPUs | `--gres=shard:rtx_3090:8192` | 8 GiB of a 3090; ample for batch 8 |
| CPUs | `8` | tokenisation / dataloading |
| RAM | `32G` | sharded dataset + Python |
| Temp disk | `tmpdisk:20480` | 20 GiB `/tmp` |

## Monitoring

```bash
squeue -u "$USER" -o "%.18i %.9P %.30j %.10T %.10M %.9l %R"
tail -f logs/<jobid>-tabula-t5.out
scancel <jobid>
sacct --format=JobID,JobName,State,ExitCode,Elapsed
```

`tail -f` on the `.out` file shows the step/loss lines and the
`--- Validation loss @ step N ---` markers.

### Notifications

Slurm mail is not used; subscribe to the WATcloud notifier instead:

```bash
./watcloud subscription <jobid> you@example.com
./watcloud subscription <jobid> --discord <webhook_url>
```

`watcloud` is on the login node only.

## One-time cluster setup

The cluster needs its own virtualenv and a copy of the processed corpus and
vocabulary; neither is in git.

```bash
ssh watcloud
mkdir -p ~/tabula_gs && cd ~/tabula_gs

python3 -m venv .venv
.venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt
```

Then copy `data/processed_guitarset/` and `vocab/guitarset.json` from the
workstation (~2 MB total):

```bash
rsync -az data/processed_guitarset vocab watcloud:~/tabula_gs/data/../ 2>/dev/null || \
rsync -az data/processed_guitarset watcloud:~/tabula_gs/data/ && \
rsync -az vocab/guitarset.json watcloud:~/tabula_gs/vocab/
```

Verify the vocabulary is the frozen GuitarSet one (size 248, hash
`6265d9bb91e68095`):

```bash
cd ~/tabula_gs && PYTHONPATH=src .venv/bin/python -c "
from tablature.tokenizer import Vocab
v = Vocab.load('vocab/guitarset.json'); print(v.size, v.hash)"
```

`torch.cuda.is_available()` is **False on login nodes** — there is no GPU
there. Only check it inside an `srun`/`sbatch` allocation:

```bash
srun --gres shard:rtx_3090:8192 --time=00:05:00 \
  ~/tabula_gs/.venv/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

The tested driver is 580.178.04 (CUDA 13.0); the cu121 wheel above works.

## Smoke test before a long run

```bash
srun --partition=compute --gres=shard:rtx_3090:4096,tmpdisk:2048 \
     --cpus-per-task=4 --mem=8G --time=00:20:00 \
     bash scripts/smoke_t5.sh
```

This runs two optimizer steps with `--metrics-every 0` (no full-track
decoding) and prints the GPU name, torch version and bf16 support first.

## Troubleshooting

**`srun: error: Unable to allocate resources: Invalid account or account/partition combination specified`**
Your WATcloud profile has no registered affiliation:

```bash
sacctmgr show user "$(whoami)"   # empty output = no account
```

**`No devices were found` / `torch.cuda.is_available() == False` inside a job**
The job did not get a GPU. Check that the GRES request is typed
(`shard:rtx_3090:8192`) and confirm the node:

```bash
scontrol show job "$SLURM_JOB_ID" | grep -E "NodeList|TresPerNode|AllocTRES"
```

**`ModuleNotFoundError: No module named 'core.schema'`**
`PYTHONPATH` is not set. Run through `scripts/train_t5.sh` (it exports
`PYTHONPATH="$REPO_DIR/src"`) or set it yourself, and use the venv interpreter,
not the system `python3`.

**Job runs much slower than expected**
You are probably sharing a GPU. `--gres=shard:rtx_3090:8192` gives 8 GiB of a
24 GiB card, so up to three tenants can share it. Check utilisation with
`nvidia-smi` inside the job. The data pipeline is not the problem: chunks are
tokenised once and cached, so the DataLoader is ~1% of step time.

Measured on `trpro-slurm1` (RTX 3090 shard, batch 8 × grad-accum 8):
32 ms per micro-batch, i.e. ~2.1 h for 30,000 optimizer steps before
evaluation overhead. The first job on a fresh workspace is ~3 s slower while
it builds the chunk cache — do not read that as steady-state throughput.

**bf16 silently disabled**
`train.py` warns and falls back to fp32. This means the node is Pascal
(GTX 1080) or the GRES was untyped. Re-request with `shard:rtx_3090:...`.
