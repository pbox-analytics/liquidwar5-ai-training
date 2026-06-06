# liquidwar deploy layer (GitOps)

This repo is the **single home** for the liquidwar RL workloads — source
(`rl/`, `coordinator.py`, `worker.py`, `simulator/`), images (`docker/` +
Kaniko jobs in `k8s/`), and now the **deploy layer**: Helm charts under
`charts/` and the ArgoCD `Application` manifests here in `argocd-apps/`.

It is fully independent of the image-gen / comfyui GitOps (which lives in
`pandoras-box-data-platform`). The two share only the k8s cluster + the one
ArgoCD instance.

## How it's wired

```
platform-root (in pandoras-box-data-platform)
  └─ liquidwar-root            # repoURL → THIS repo, path: argocd-apps
       ├─ liquidwar-coordinator   → charts/liquidwar-coordinator
       ├─ liquidwar-cpu-worker    → charts/liquidwar-cpu-worker
       └─ liquidwar-gpu-trainer   → charts/liquidwar-gpu-trainer
```

`liquidwar-root` stays in the data-platform repo (so the cluster's app-of-apps
bootstraps it), but it points *here* — so every liquidwar deployable is
version-controlled alongside its source.

## Before this goes live

1. **Confirm the branch:** these manifests target `master`. If you rename the
   default branch to `main`, update `targetRevision` here and in
   `liquidwar-root`.
2. **Repo must be readable by ArgoCD:** transfer this repo into the
   `pbox-analytics` org (the existing org credential then covers it).
3. **Cut over:** point `liquidwar-root` at this repo and remove the
   `charts/liquidwar-*` + `argocd-apps/liquidwar/` copies from data-platform.

The Helm charts were ported as-is from data-platform (proven, working). Convert
them to your `k8s/`-style raw manifests later if you prefer — only the
`Application` `path`/source would change.
