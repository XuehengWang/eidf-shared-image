# EIDF shared CPU/GPU research image (experimental)

## Compatibility first
EIDF A100 pods previously reported NVIDIA driver **570.124.06**, which fits CUDA 12.8 but not the normal CUDA 13 minimum driver of 580. SGLang **0.5.20** upstream currently uses CUDA 13, whereas its **0.5.19** release was the last officially maintained CUDA-12 lane. This Dockerfile **attempts** CUDA 12.8 + Python 3.12 + PyTorch 2.10.0 + SGLang 0.5.20 + kvcached 0.1.6. Installation/runtime compatibility is **unverified**. Treat any image build failure or GPU kernel failure as a real blocker; do not weaken the cluster's driver isolation to work around it. Check which kernel wheels and dependencies work on your EIDF A100 nodes before relying on the stack.

## Files
- `Dockerfile`: image, Python/Conda, tools, pinned packages.
- `dev-job.yaml`: CPU-only Kueue Job; same image and `xh-gpu-research` PVC.
- `gpu-job.yaml`: 2x A100 Job; same image and PVC.
- `.github/workflows/build.yml`: builds image from GitHub Actions and pushes to GHCR.

## Build without consuming EIDF GPUs
1. Create your own GitHub repository (e.g. `eidf-llm-image`), commit this entire directory, and push it.
2. GitHub Actions > **Build and publish EIDF image** > **Run workflow**, or trigger with a push to Dockerfile.
3. Confirm the workflow has completed successfully. The Python / CUDA dependency resolver can fail; success is NOT guaranteed.
4. Make the resulting GHCR package public (GitHub package settings), or arrange an EIDF imagePullSecret for a private package.
5. In **both** `dev-job.yaml` and `gpu-job.yaml`, replace `REPLACE_WITH_YOUR_GITHUB_USERNAME` with your GitHub account/org name (lowercase in GHCR paths).

Alternative: build and push with Docker Buildx on a trusted Linux amd64 host:

    docker buildx build --platform linux/amd64 \
      -t ghcr.io/USERNAME/eidf-llm-dev:cuda12.8-v1 \
      --push .

## Deploy using eidf029-host26

    kubectl get pvc xh-gpu-research -n eidf029ns
    kubectl create -f dev-job.yaml -n eidf029ns
    kubectl get jobs -n eidf029ns -l eidf/user=s2522871-infk8s
    kubectl get pods -n eidf029ns | grep xh-dev
    kubectl exec -it POD_NAME -n eidf029ns -- bash

Inside the dev pod:

    which python; python -V
    conda --version
    python -c 'import torch,sglang,kvcached; print(torch.__version__, torch.version.cuda)'
    cd /workspace

When you need GPUs:

    kubectl create -f gpu-job.yaml -n eidf029ns
    kubectl get pods -n eidf029ns | grep xh-gpu-test
    kubectl exec -it POD_NAME -n eidf029ns -- bash

Inside GPU pod, verify the driver and imports BEFORE running a server:

    nvidia-smi
    python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
    python -m pip check

By default, kvcached autpatching is DISABLED until GPU runtime checks pass. Enable for a specific experiment:

    export ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1

## Extending later
Edit Dockerfile, build a **new tag** (e.g. cuda12.8-v2), update both YAMLs. Rebuilding the image avoids environment drift. Never store Hugging Face tokens or GHCR passwords in YAML or Dockerfile. Use authorised Kubernetes secrets instead.

## Alternative if GPU kernels fail
Ask EIDF which supported driver is deployed on GPU nodes and whether an R580+ driver is available. If not, use SGLang 0.5.19 on a verified CUDA-12 stack, but note this would NOT meet the requested exact SGLang version; consult the kvcached compatibility matrix before downgrading.
