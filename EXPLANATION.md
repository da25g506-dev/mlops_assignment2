# Explanation

## Repository

- **GitHub repo:** https://github.com/da25g506-dev/mlops_assignment2
- **Final PR (Kubernetes manifests + end-to-end validation on a live cluster):** https://github.com/da25g506-dev/mlops_assignment2/pull/4
- **Release PR (develop -> main):** https://github.com/da25g506-dev/mlops_assignment2/pull/5

## What was built

A CIFAR-10 image classifier (ResNet-18, CIFAR-adapted stem) with a PyTorch
training script, a FastAPI inference service, multi-stage Docker images for
both, and Kubernetes manifests to run training as a `Job` and serving as a
2-replica `Deployment` with liveness/readiness probes, a `Service`, and an
`HPA`. Everything was validated for real: both images were built and run
locally with mounted volumes, and the full manifest set was applied to a
live local `kind` cluster, ending with a `kubectl port-forward` + `curl
/predict` round trip against pods running on that cluster.

For a component-by-component deep dive of every file and design decision,
see [`WALKTHROUGH.md`](WALKTHROUGH.md); for a Q&A-style study companion,
see [`STUDY_GUIDE.md`](STUDY_GUIDE.md).

## Full training run

Beyond the fast verification runs used to prove the pipeline's plumbing
works end-to-end, the shipped default config
(`configs/training_config.yaml`: ResNet-18, 10 epochs, batch size 64, the
**full** 50,000-image CIFAR-10 train set, `subset_fraction: 1.0`) was run
to completion for real via `docker run` with the training image and
mounted volumes (see `README.md` for the exact command). It ran for
**~7 hours** on this CPU-only host — slower than a typical CPU CIFAR-10
run both because MKL-DNN is disabled (see below) and because it is
genuinely training on all 50,000 images for the full 10 epochs, not a
subset. All 10 epochs completed without early stopping ever triggering;
the best checkpoint (lowest validation loss) was saved at epoch 8,
reaching **86.79% validation accuracy** (val_loss 0.3947); full per-epoch
metrics are in `README.md`. Serving that checkpoint against 6 held-out
CIFAR-10 test images (one per class, never seen during training)
correctly classified 5 of 6 with high confidence (>99% for cat, ship,
frog, and truck); the one miss — an airplane image predicted as "ship" —
is a well-known CIFAR-10 confusion pair (both are often photographed
against a plain sky/water background at a similar angle) and is
consistent with a model at 86.79% accuracy rather than a defect in the
pipeline.

## Most challenging part

The most challenging issue wasn't in the application code — it was two
environment-level bugs that only appeared once the code left a "happy path"
environment.

First, PyTorch's convolution op reliably crashed with `SIGFPE` (no Python
traceback, just exit code 136) on this machine's AMD EPYC CPU, but only
inside actual `Conv2d` forward passes — plain tensor ops were fine. I
isolated it by stepping through ResNet-18 layer by layer with flushed
prints until the crash localized to the first conv layer, then reproduced
it with a bare `nn.Conv2d` on a synthetic tensor to rule out anything
model-specific. Several documented workarounds for this exact class of bug
(`MKL_ENABLE_INSTRUCTIONS`, `MKL_DEBUG_CPU_TYPE`, `ATEN_CPU_CAPABILITY`
environment variables) did nothing; the actual fix was disabling the
MKL-DNN backend entirely (`torch.backends.mkldnn.enabled = False`), a
oneDNN CPU-dispatch bug affecting some AMD EPYC hosts. This is now a
one-line guard at the top of `src/model.py` so it applies to every entry
point automatically.

Second, when the training `Job` ran on the `kind` cluster, it appeared to
hang for 20+ minutes with no progress, despite behaving fine locally in
plain Docker. Checking the pod's cgroup `cpu.stat` showed ~99% of CPU
periods throttled. The cause: PyTorch sizes its intra-op thread pool to the
number of CPUs the *host* exposes (16, on this machine), completely
ignoring the pod's CFS quota (`cpu: 2` in the Job spec) — so 16 threads
were contending for 2 cores' worth of scheduler time. Pinning
`OMP_NUM_THREADS`/`MKL_NUM_THREADS` to match the CPU request in
`training-job.yaml` dropped throttling to ~33% and the same job completed
in under 3 minutes. It's a good illustration of why resource requests
alone aren't sufficient for CPU-bound Python/native workloads in
Kubernetes — the process itself has to be told about the quota.

A smaller but real issue was Docker image bloat: PyPI's default `torch`
wheel bundles the full CUDA runtime, ballooning the training image to
5.1GB. Switching both requirements files to the `cpu`-only wheel index
(`--extra-index-url https://download.pytorch.org/whl/cpu`) cut that to
1.1GB with no behavior change on CPU-only hosts like this one.
