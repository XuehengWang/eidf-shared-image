# EIDF CUDA 13.0 shared development image

Use the **same custom CUDA image** for CPU-only editing and GPU experiments, mounting
`xh-gpu-research` at `/workspace` in both jobs.

## Versions
- Base: `nvidia/cuda:13.0.2-devel-ubuntu24.04` (linux/amd64)
- Python 3.12 (Miniforge), Vim, Git, GCC/G++, CMake, NVCC
- PyTorch 2.13.0 (CUDA 13.0) and torchvision 0.28.0
- SGLang 0.5.20
- kvcached 0.1.6
- Additional: IPython, pytest, psutil, pandas

## Build and publish (no Docker required locally)

Commit all project files to a GitHub repository under `xuehengwang` and trigger
**Actions → Build and publish EIDF image**. GitHub Actions builds linux/amd64 and
publishes:

`ghcr.io/xuehengwang/eidf-llm-dev:cuda13.0-v1`

Ensure the GHCR package is publicly pullable, or configure Kubernetes
`imagePullSecrets` for EIDF. The workflow makes the GHCR owner lowercase.

The build performs pip dependency and metadata version checks. It does **not**
prove that SGLang kernels, kvcached, or GPU execution work on EIDF.

## Start CPU-only development Job

```bash
kubectl create -n eidf029ns -f dev-job.yaml
kubectl get pods -n eidf029ns | grep xh-dev
kubectl exec -it -n eidf029ns <DEV_POD_NAME> -- bash
cd /workspace
python --version
```

## Start GPU Job

```bash
kubectl create -n eidf029ns -f gpu-job.yaml
kubectl get pods -n eidf029ns | grep xh-gpu-test
kubectl exec -it -n eidf029ns <GPU_POD_NAME> -- bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

EIDF worker previously reported NVIDIA driver 570.124.06, which is too old
for regular CUDA 13.0 execution; NVIDIA typically requires an R580+ driver
for CUDA 13.0. The image can be built and used for CPU development without
GPU allocation; GPU jobs may fail until a compatible driver is available.

## Extending the image

Add further `RUN python -m pip install ...` steps near the end of the Dockerfile,
then tag a new version (e.g. `cuda13.0-v2`) and update both Job manifests.
Do not install packages only into the running container if you want them to
appear in future pods. Source code, results, and any files stored in the
mounted `/workspace` PVC persist independently of the image.

## EIDF scheduling

The jobs use Kueue and a 900-second active execution deadline. You may have
to wait for admission. Both jobs request CPU and RAM; only `gpu-job.yaml`
requests 2× A100 GPUs. Delete finished/expired Jobs to avoid clutter.

## Notes

- Verify exact A100 node product labels in your cluster.
- kvcached automatic patching is disabled by default (`ENABLE_KVCACHED=false`,
  `KVCACHED_AUTOPATCH=0`); enable only when you deliberately test it.
- This is an **untested build recipe**. GitHub-hosted runners may run out of
  disk space when resolving the large ML dependency stack.
