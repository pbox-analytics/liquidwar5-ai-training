# GitOps Deployment Design — Liquid War 5 AI Training

_2026-05-30 — revised after verifying the existing cluster. Piggybacks on the existing
pandora GitOps stack; does NOT build new infrastructure._

## Goal

Make Liquid War 5 AI training reproducible and consistent across the whole fleet without
mutating host systems, so a `git push` rolls out fleet-wide via the **existing** ArgoCD setup
instead of hand-running `ssh + apt + make` on each machine.

Two coupled repos:
- **`liquidwar5-ai`** — C game engine (autotools, SDL2). Produces `src/liquidwar` + `data/liquidwar.dat`.
- **`liquidwar5-ai-training`** — Python/uv trainer (this repo). GA + GPU sim, distributed via Kafka.

## Why containerize (motivation from real failures)

Every friction point hit while bringing the fleet to a common commit traces to **host drift**:

| Problem observed | Root cause | Container fixes it by |
|---|---|---|
| Build needed `apt install` on swolf + pbox | Deps not on hosts | Deps baked into image; hosts stay pristine |
| pbox build failed: undefined `fixsqrt`/`_cos_tbl`/`rest` | **Stale `.o` from the Allegro era** | Fresh build context every time |
| swolf "built" but linked stale Allegro | Leftover `liballeg.so` in `/lib` | Clean base image, no leftovers |
| `pkill Xvfb` collided with ComfyUI on shared GPU host | Shared PID/display namespace | Isolated PID namespace per pod |
| gcc 13 (swolf) vs gcc 15 (pbox) | Different OS releases | One pinned toolchain in the image |

**Verified:** the engine at `d38b749` builds clean on **both** arm64 (swolf) and amd64 (pbox)
after `make clean`, against the SDL2 shim with **no Allegro present**. The SDL2 migration is
complete; the earlier "incomplete migration" symptom was purely stale objects.

## The existing platform we piggyback on (verified 2026-05-30)

Single **k3s** cluster, 5 nodes (ptow control-plane + pbox/tank/storm/swolf workers). Already running:

| Capability | What exists | How liquidwar uses it |
|---|---|---|
| **ArgoCD** (ns `argocd`) | app-of-apps `platform-root`, 40+ apps Synced/Healthy | Add one app entry → commit = rollout |
| **Registry** | `pandoras-box.local:5000` (push) / `registry.homer-ai.svc.cluster.local:5000` (pull); TLS self-signed, pinned to pbox, 18TB on `/mnt/dlred1` | Push the worker image here |
| **Kafka** | Strimzi-operated, ns `kafka`, + schema-registry | The trainer's job/result queue already in-cluster |
| **GPU** | gpu-operator + `nvidia-device-plugin` ArgoCD app, dcgm-exporter | gpu-evolve scheduling already solved |
| **dispatcher-worker** | per-node distributed worker, GitOps-managed via ApplicationSet | **The exact template to copy** |

GitOps repos (Helm charts, app-of-apps under `argocd-apps/`):
- `pbox-analytics/pandoras-box-data-platform` — platform services (kafka, spark, jupyter, minio, monitoring, …). **← liquidwar goes here (user decision).**
- `pbox-analytics/project-homer` — AI/compute workloads (comfyui, dispatcher-*, registry).
- `pbox-analytics/pandoras-box-data-acquisition` — kafka topics (branch `dev`).

### The dispatcher pattern (our model)
project-homer implements `dispatcher-submitter` + per-node `dispatcher-worker-{ptow,pstorm,pbox-0/1/2,tank-a/b}` + `redis-dispatcher`. The per-node fan-out is an **ArgoCD ApplicationSet** (`argocd-apps/dispatcher-worker-set.yaml`, generated from `gpus.yaml` via `scripts/regen-gpu-values.py`) whose `list` generator injects per-node facts (node hostname, GPU UUID, vram) into a Helm `valuesObject`. Each child Application pins to its node via `nodeSelector: kubernetes.io/hostname`.

This is structurally identical to liquidwar training: a coordinator hands out parameter sets, per-node workers run games, results flow back. **We copy this shape.** The one difference: liquidwar's trainer uses **Kafka** (already in-cluster, ns `kafka`) for jobs/results, whereas dispatcher-worker uses **Redis streams** — so our worker consumes Kafka instead.

## Image build & push (in-cluster build Job + ArgoCD Image Updater — user decision 2026-05-30)

**Why not the homer manual flow:** the existing convention (human SSHes to ptow → `docker build` → `docker push` → hand-edit tag in `values.yaml` → commit) works but isn't process-oriented — it depends on one host's docker config and a human remembering to build-from-clean and bump the tag.

**Note on the "NFS push" idea:** distributing images via an NFS-shared tarball (`docker save`/`load` per host) is *worse* than what we have — no layer dedup, no caching, full copy to every node. The **registry already IS the repeatable distribution layer**: content-addressed, layer-deduplicated, pulled on-demand by each node's containerd and verified by digest. (Its backing store happens to be an 18TB disk on pbox, the NFS server — so bytes do live on pbox, but the abstraction on top is the registry protocol, not a mount. Keep it.) The genuine wart the NFS idea was reacting to is the multi-arch `ssh save|load` tarball hop — fixed by having both arch builders push to the registry directly, not by adding NFS.

**Chosen pipeline (both greenfield — neither installed yet):**
1. **In-cluster build Job** (Kaniko or BuildKit) — builds the image in a clean pod and pushes to the registry. No host docker, no host drift, reproducible every run. Registry has TLS but no auth (LAN-trusted), so the Job needs only CA trust + the `pandoras-box.local:5000` address — no registry creds. Triggered by a `kubectl create job`/`Job` template (later: on git push via a webhook or a CronJob that watches the repo).
2. **ArgoCD Image Updater** — watches the registry for new tags and auto-bumps the chart's image tag via **git write-back** to `pandoras-box-data-platform`, so ArgoCD then rolls it out. Removes the manual `values.yaml` edit → **git push is the only human action.** Needs git creds to the manifests repo; uses a semver/newest-build update strategy against the immutable version tags.

Tags remain **immutable** (never reuse; no `latest`; `imagePullPolicy: IfNotPresent`).

**Multi-arch caveat:** Kaniko has no in-pod QEMU — native multi-arch = one build Job per arch (amd64 on an amd64 node, arm64 on swolf) + a manifest-combine step. arm64 only matters for swolf/air, so ship an **amd64-only Job first** and add arm64 later.

**Two images:**
- **`liquidwar-cpu-worker`** — headless C engine + SDL2 runtime libs + xvfb + python + `worker.py`. Bulk of the ~92-core fleet. Built by extending the engine's existing `misc/docker/Dockerfile-server.in` (already a multi-stage SDL2 build) to also build `liquidwar` (not just `liquidwar-server`) and add the python/training layer.
- **`liquidwar-gpu-evolve`** — torch/CUDA + `gpu_evolve.py` + `simulator/`. No C engine (GPU sim is pure torch). GPU-scheduled.

Built by an **in-cluster Kaniko/BuildKit Job** (see "Image build & push" above), one Job per arch pushing to `pandoras-box.local:5000/pbox/liquidwar-cpu-worker:<ver>-<arch>`, combined into a manifest list. amd64-first.

## Workloads

- **cpu-worker** — `charts/liquidwar-cpu-worker` (model on dispatcher-worker). One Deployment per
  node (ApplicationSet or DaemonSet), `--workers` = node cores, consumes Kafka jobs. No GPU.
- **gpu-evolve** — `charts/liquidwar-gpu-evolve`, requests `nvidia.com/gpu: 1`, scheduled to GPU
  nodes. **Verify per-node GPU allocatable before pinning** (re-check the earlier unverified
  "pbox allocatable=0" claim).
- **coordinator** — single replica, GA breeding, anywhere.
- Headless/Xvfb is safe in-pod (isolated PID namespace; engine auto-starts `Xvfb :99`).

## Plan placement (user decisions 2026-05-30)
- Chart + ArgoCD app live in **`pandoras-box-data-platform`** (`charts/liquidwar-*` + an entry under `argocd-apps/`).
- Image push reuses the **manual ptow-build → `pandoras-box.local:5000` → bump values.yaml tag → commit** flow.

## Open questions / to verify
1. **Build context spanning two repos** (engine + training). Options: submodule, a `build/` dir
   cloning both at a pinned ref, or buildx `--build-context`. (Lean: pinned-ref checkout dir.)
2. **uv vs pip** in the image — repo has `uv.lock`; prefer `uv sync` for fidelity.
3. **Per-node GPU allocatable** — confirm which nodes expose `nvidia.com/gpu` for gpu-evolve
   (re-verify; do NOT trust the earlier unconfirmed pbox=0 note).
4. **Kafka topics** — are the liquidwar topics (`ml.liquidwar5.*`) defined in the
   data-acquisition `kafka/topics` GitOps path, or created ad hoc? Add them to GitOps.
5. **CPU/GPU sim parity** — the GPU sim is a separate reimplementation of the C engine; add a
   parity test (same seed → compare outcome) to prevent drift. Recommended.
6. **Coordinator/results** — the trainer's coordinator + Kafka result topic: deploy coordinator
   in-cluster too, or run it ad hoc? (Lean: in-cluster for full GitOps.)

## Phased rollout
- **Phase 1 — Image:** extend `Dockerfile-server.in` → headless cpu-worker; build amd64 on ptow;
  run one container against the in-cluster Kafka to prove it consumes a job and emits a result.
- **Phase 2 — Chart + GitOps:** `charts/liquidwar-cpu-worker` in data-platform + an ArgoCD app;
  commit; confirm ArgoCD deploys one worker; scale per-node.
- **Phase 3 — GPU + coordinator:** add `liquidwar-gpu-evolve` (GPU-scheduled) + coordinator;
  wire Kafka topics into GitOps; add the parity test.
- **Phase 4 — Multi-arch + cleanup:** add the arm64 build for swolf/air; retire the old
  SSH/`launch_all.sh` flow and the `CHANGEME` `machines.json`.
