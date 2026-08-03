# mlops-pytorch-pipeline

An end-to-end MLOps pipeline for a CIFAR-10 image classifier: PyTorch training
and inference code, multi-stage Docker images, and Kubernetes manifests to
run training as a Job and serve predictions behind an autoscaled Deployment.

Built for Assignment 2 ("Deploying PyTorch ML Workloads with Docker &
Kubernetes") of the MLOps & Infrastructure for Machine Learning course. See
[`EXPLANATION.md`](EXPLANATION.md) for a write-up on design decisions and
challenges encountered while building this.

## Architecture

```
                         ┌─────────────────────────┐
                         │   configs/*.yaml          │
                         │   (hyperparameters)       │
                         └────────────┬──────────────┘
                                      │ mounted at /app/configs
                                      ▼
 ┌──────────────┐   docker build   ┌──────────────────────┐   torch.save()   ┌────────────────────┐
 │ src/model.py  │────────────────▶│  docker/Dockerfile.train│─────────────────▶│ checkpoints/         │
 │ src/dataset.py│                 │  -> mlops-train:v1      │                  │  classifier_v1.pt    │
 │ src/train.py  │                 └──────────────────────┘                  └──────────┬──────────┘
 └──────────────┘                                                                        │
                                                                                          │ mounted read-only
                                                                                          ▼
 ┌──────────────┐   docker build   ┌──────────────────────┐   GET /health     ┌────────────────────┐
 │ src/model.py  │────────────────▶│  docker/Dockerfile.serve│──────────────────│  FastAPI service     │
 │ src/dataset.py│                 │  -> mlops-serve:v1      │   POST /predict  │  (uvicorn :8080)     │
 │ src/serve.py  │                 └──────────────────────┘◀─────────────────└────────────────────┘
 └──────────────┘

  Kubernetes (namespace: ml-training)
  ┌───────────────────────────────────────────────────────────────────────────────────┐
  │  ConfigMap            Job                      Deployment (2 replicas)  Service    │
  │  training-config ───▶ cifar10-training-job ──▶  model-serving  ◀──────  model-serving│
  │  (training_config      │        │                  │  liveness  /health  (ClusterIP  │
  │   .yaml)                │        │                  │  readiness /health   :80→8080) │
  │                         ▼        ▼                  ▼                               │
  │                 training-data-pvc  model-checkpoints-pvc (shared, read-only in Deploy)│
  │                                                       ▲                              │
  │                                              HorizontalPodAutoscaler                 │
  │                                              (CPU target 70%, 2-5 replicas)           │
  └───────────────────────────────────────────────────────────────────────────────────┘
```

## Repository layout

```
.
├── src/
│   ├── model.py       # ResNet-18 (CIFAR-adapted) + a lightweight SimpleCNN
│   ├── dataset.py      # CIFAR-10 DataLoaders with train/eval transforms
│   ├── train.py        # Training loop: config -> train -> checkpoint, JSON-line logs
│   └── serve.py         # FastAPI inference service (/health, /predict)
├── configs/
│   ├── training_config.yaml          # shipped defaults (10 epochs, full dataset)
│   └── training_config.verify.yaml   # fast override for local/demo verification
├── docker/
│   ├── Dockerfile.train   # multi-stage, CPU-only torch wheels, training entrypoint
│   └── Dockerfile.serve   # multi-stage, non-root user, HEALTHCHECK, port 8080
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── training-job.yaml        # Job + PVCs for /app/data and /app/checkpoints
│   ├── training-job-gpu.yaml    # bonus: GPU-scheduled variant
│   ├── serving-deployment.yaml  # 2 replicas, probes, rolling update
│   ├── serving-service.yaml     # ClusterIP :80 -> :8080
│   └── hpa.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
├── tests/
│   └── test_model.py
└── .github/workflows/ci.yml
```

## Setup

### Prerequisites

- Docker
- Python 3.11 (only needed for running outside Docker)
- `kubectl` and `kind` (only needed for the Kubernetes workflow)

### Local Python environment (optional)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements/train.txt -r requirements/serve.txt
```

> Both requirement files pin CPU-only PyTorch wheels
> (`--extra-index-url https://download.pytorch.org/whl/cpu`) so installs and
> Docker images stay small; PyTorch will simply use the CPU where no GPU is
> present.

### Run tests

```bash
pytest tests/ -v
```

## Docker workflow

**Build the training image:**

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
```

**Run training with mounted volumes** (config, data cache, and checkpoint
output all live on the host so results persist across runs):

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  -v "$(pwd)/configs/training_config.yaml:/app/configs/training_config.yaml:ro" \
  mlops-train:v1
```

Training emits JSON-line events to stdout (`config_loaded`,
`device_selected`, `epoch_complete`, `checkpoint_saved`,
`early_stopping`/`training_complete`) and writes the best checkpoint to
`checkpoints/classifier_v1.pt`.

### Full training run (real evidence, not a smoke test)

The shipped `configs/training_config.yaml` (ResNet-18, 10 epochs, batch
size 64, full 50,000-image CIFAR-10 train set, `subset_fraction: 1.0`) was
run to completion with `docker run` exactly as shown above — a genuine
full training run, not the fast verification config. It ran for
**~7 hours** on this CPU-only host (MKL-DNN disabled — see
[`EXPLANATION.md`](EXPLANATION.md) — and no early stop, since validation
loss never failed to improve for 3 consecutive epochs). Final metrics:

| Epoch | Train loss | Train acc | Val loss | Val acc | Checkpoint saved? |
|------:|-----------:|----------:|---------:|--------:|:------------------:|
| 1 | 1.3576 | 50.56% | 1.1535 | 60.97% | ✅ |
| 2 | 0.9017 | 68.27% | 1.0076 | 65.33% | ✅ |
| 3 | 0.7100 | 75.18% | 1.1051 | 65.74% | |
| 4 | 0.5916 | 79.62% | 0.6320 | 79.17% | ✅ |
| 5 | 0.5168 | 82.14% | 0.5296 | 82.88% | ✅ |
| 6 | 0.4574 | 84.22% | 0.5266 | 82.18% | ✅ |
| 7 | 0.4071 | 86.02% | 0.4987 | 84.03% | ✅ |
| **8** | **0.3722** | **87.15%** | **0.3947** | **86.79%** | ✅ **(final checkpoint)** |
| 9 | 0.3376 | 88.32% | 0.4281 | 85.71% | |
| 10 | 0.3108 | 89.17% | 0.3997 | 86.91% | |

The best-validation-loss checkpoint (epoch 8, 86.79% val accuracy) is what
`classifier_v1.pt` contains — epochs 9-10 trained further and reached
slightly higher *accuracy* but higher *loss*, so per the save policy
("save only when val_loss improves") they didn't overwrite the epoch-8
checkpoint. Serving that checkpoint against 6 real CIFAR-10 test images
(one per class, held out from training) correctly predicted 5/6 with high
confidence (>99% for cat, ship, frog, truck; 99.9% for automobile); the
one miss was an airplane image predicted as "ship" — a well-known
CIFAR-10 confusion pair, and consistent with an 86.79%-accuracy model.

See [`WALKTHROUGH.md`](WALKTHROUGH.md) for a full explanation of every
component and [`STUDY_GUIDE.md`](STUDY_GUIDE.md) for a Q&A-style study
guide covering the whole project.

**Build the serving image:**

```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

**Run the serving container**, mounting the checkpoint produced above:

```bash
docker run --rm -p 8080:8080 \
  -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
  mlops-serve:v1
```

**Test it:**

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Kubernetes workflow

This project targets any standard cluster; it was validated end-to-end on a
local [`kind`](https://kind.sigs.k8s.io/) cluster.

```bash
# 1. Create a local cluster and load the images built above
kind create cluster --name mlops-assignment
kind load docker-image mlops-train:v1 --name mlops-assignment
kind load docker-image mlops-serve:v1 --name mlops-assignment

# 2. Apply the training layer
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml

# 3. Wait for the training Job to complete, then deploy serving
kubectl wait --for=condition=complete job/cifar10-training-job -n ml-training --timeout=1200s
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# 4. Verify
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

# 5. Test the prediction endpoint
kubectl port-forward svc/model-serving 8080:80 -n ml-training &
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

The training Job mounts the `training-config` ConfigMap at `/app/configs`
and two PVCs (`training-data-pvc` at `/app/data`, `model-checkpoints-pvc` at
`/app/checkpoints`). The serving Deployment mounts the same checkpoints PVC
**read-only**, so the pipeline is: train once on the cluster → serve the
resulting checkpoint from as many replicas as needed.

`k8s/training-job-gpu.yaml` is a bonus variant of the training Job for
clusters with GPU node pools: it requests `nvidia.com/gpu: 1`, adds a
`nodeSelector` for GPU-labeled nodes, and tolerates the standard
`nvidia.com/gpu=present:NoSchedule` taint.

## Git workflow

Work is organized as `main` ← `develop` ← stacked feature branches, merged
via pull requests with [Conventional Commits](https://www.conventionalcommits.org/)
messages:

- `feature/repo-scaffold` — project structure, `.gitignore`, CI skeleton
- `feature/pytorch-model` — model/dataset/train/serve implementation
- `feature/docker-training` — Dockerfiles, pinned requirements, verified locally
- `feature/k8s-deployment` — Kubernetes manifests, validated on a live `kind` cluster

## CI

`.github/workflows/ci.yml` runs on every push/PR to `main`/`develop`:

1. **lint-and-test**: `ruff check` + `pytest tests/`
2. **docker-build**: builds both `Dockerfile.train` and `Dockerfile.serve`
   (gated on lint-and-test passing), with GitHub Actions layer caching
