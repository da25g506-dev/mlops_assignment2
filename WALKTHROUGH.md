# Walkthrough: everything in this repo, explained

This document exists so you can explain, defend, and reason about every
piece of this project without having to re-derive it from the code. It
walks the pipeline in the order data flows through it: model → data →
training → serving → containers → Kubernetes → CI/CD → git workflow. Each
section covers *what* exists, *why* it's built that way, and *what would
break* if it weren't. Pair this with [`STUDY_GUIDE.md`](STUDY_GUIDE.md),
which turns this same material into a question-and-answer format for
self-testing.

---

## 1. The problem being solved

CIFAR-10 is a 60,000-image dataset (50,000 train / 10,000 test), 32×32
pixels, RGB, 10 mutually-exclusive classes: airplane, automobile, bird,
cat, deer, dog, frog, horse, ship, truck. The task: train an image
classifier, then serve it behind an HTTP API, then run both the training
job and the serving API on Kubernetes with production-shaped concerns
(health checks, autoscaling, resource limits, rolling updates).

This is not a research project chasing state-of-the-art accuracy — it's an
**infrastructure/MLOps exercise**. The model architecture is intentionally
a well-understood, "textbook" choice (ResNet-18) so that all the
interesting engineering is in the packaging, deployment, and operational
correctness, not in novel modeling.

---

## 2. `src/model.py` — the model

Two architectures are implemented behind one factory function:

```python
def get_model(architecture: str, num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    architecture = architecture.lower()
    if architecture == "resnet18":
        return _resnet18_for_small_images(num_classes=num_classes, pretrained=pretrained)
    if architecture == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model architecture: {architecture}")
```

**Why a factory function instead of just importing `resnet18` directly?**
Because `train.py` and `serve.py` both need to construct "whatever
architecture is named in the config/checkpoint" without hardcoding a
class. `train.py` reads the architecture name from
`training_config.yaml`; `serve.py` reads it back out of the checkpoint
dict it loaded (`checkpoint["architecture"]`) so serving always
reconstructs the *exact* architecture that was trained, even if the config
file has since changed. This makes checkpoints self-describing.

### 2a. ResNet-18, adapted for 32×32 images

```python
def _resnet18_for_small_images(num_classes, pretrained=False):
    weights = "IMAGENET1K_V1" if pretrained else None
    model = resnet18(weights=weights)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

Stock `torchvision.models.resnet18` was designed for **224×224 ImageNet**
images. Its stem is a 7×7 stride-2 convolution followed by a 3×3
stride-2 max-pool — by design, that stem alone downsamples the input by
4× *before* any residual block even sees it. Applied to a 32×32 CIFAR
image, that stem would shrink it to 8×8 in the first few layers, throwing
away most of the already-tiny spatial resolution before the network gets
a chance to learn anything from it.

The fix (a standard, widely-used adaptation for CIFAR-scale ResNets, not
something invented for this project):
- Replace the 7×7 stride-2 stem conv with a 3×3 **stride-1** conv — no
  downsampling at the very first layer.
- Replace the stem's max-pool with `nn.Identity()` (a no-op) — removes the
  second 2× downsample.
- Replace the final fully-connected layer so its output dimension matches
  `num_classes` (10 for CIFAR-10) instead of ImageNet's 1000.

Everything else about ResNet-18 — the four stages of residual blocks, the
skip connections, batch norm, global average pool — is untouched.

**Why `pretrained` defaults to `False`:** ImageNet-pretrained weights
assume the *original* stem and 224×224 inputs. Since the stem is being
replaced, the pretrained conv1 weights would be discarded anyway, and
using ImageNet-pretrained weights for the *rest* of the network on 32×32
inputs is a debatable transfer-learning choice this assignment doesn't
require — so the code defaults to training from scratch, and only takes
the ImageNet weights if the caller explicitly opts in via
`pretrained=True`.

### 2b. `SimpleCNN` — a lightweight alternative

A plain 3-conv-block CNN (32→64→128 channels, each block:
conv→batchnorm→ReLU→maxpool), flattened and passed through a 2-layer
classifier head with dropout. It exists as a much smaller/faster fallback
architecture — useful for quick smoke tests or CPU-constrained
environments — selectable via the same `get_model()` factory by setting
`architecture: simplecnn` in the config. It is not used for the shipped
default run, but the assignment's design allows swapping architectures
without touching `train.py`/`serve.py`, and this proves that
extensibility actually works (it's also exercised in the parametrized
unit tests).

### 2c. The MKL-DNN guard at the top of the file

```python
torch.backends.mkldnn.enabled = False
```

This line runs at **import time**, before any model is constructed. It
was added after debugging a real crash (see §9 "Bugs found and fixed" for
the full story): on this machine's AMD EPYC CPU, PyTorch's MKL-DNN
(oneDNN) backend has a CPU-dispatch bug that causes `Conv2d` forward
passes to crash with `SIGFPE` — no Python exception, no traceback, the
process just dies with exit code 136. Disabling MKL-DNN routes
convolutions through a different, slower-but-correct compute path.
Because this line lives in `model.py` and runs on import, *every* entry
point that imports the model (`train.py`, `serve.py`, the unit tests) gets
the fix automatically — there's no way to accidentally construct a model
without it.

---

## 3. `src/dataset.py` — data loading

```python
def get_dataloaders(data_dir, batch_size=64, num_workers=2, subset_fraction=1.0):
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=get_transforms(train=True))
    val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=get_transforms(train=False))
    train_dataset = _maybe_subset(train_dataset, subset_fraction)
    val_dataset = _maybe_subset(val_dataset, subset_fraction)
    ...
```

Uses `torchvision.datasets.CIFAR10` directly rather than a hand-rolled
loader — no reason to reimplement a well-tested, widely-used dataset
loader. `download=True` means: if the CIFAR-10 tarball isn't already at
`data_dir`, torchvision fetches and extracts it there; if it is already
present (e.g. from a mounted volume with a prior download), it's reused
with no network call. This is exactly what makes the Docker/Kubernetes
volume-mounting story work: mount a host directory or PVC at `/app/data`,
and the *first* run downloads once, every subsequent run reuses the cache.

### 3a. Transforms — why training and eval differ

```python
def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
```

Training gets **data augmentation** (random horizontal flip, random
4-pixel-padded crop) — these are standard CIFAR-10 augmentations that
create synthetic variation so the model doesn't just memorize exact pixel
positions, improving generalization. Evaluation/inference gets **no
augmentation** — you want a deterministic, reproducible transform of the
real input when you're measuring accuracy or serving a prediction, not a
randomly perturbed one. Both paths share the same normalization constants
(`CIFAR10_MEAN`/`CIFAR10_STD`, the standard precomputed per-channel
mean/std for this dataset) — normalization must be identical between
train and inference, or the model sees a distribution shift at serving
time that it never saw during training.

`serve.py` imports and reuses this exact same `get_transforms(train=False)`
function — meaning the inference path is *guaranteed* to preprocess images
identically to how validation images were preprocessed during training.
There's no second, hand-copied preprocessing implementation that could
drift out of sync.

### 3b. `subset_fraction` — the fast-verification knob

```python
def _maybe_subset(dataset, fraction):
    if fraction >= 1.0:
        return dataset
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"subset_fraction must be in (0, 1], got {fraction}")
    subset_size = max(1, int(len(dataset) * fraction))
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size]
    return Subset(dataset, indices.tolist())
```

This is the mechanism behind `configs/training_config.verify.yaml`
(`subset_fraction: 0.05`) versus the shipped `configs/training_config.yaml`
(`subset_fraction: 1.0`). With a fixed seed (`42`), the subsample is
**deterministic and reproducible** — the same 5% slice every time,
rather than a different random sample on every run, which matters for
comparing verification runs to each other. When `fraction >= 1.0` the
dataset is returned untouched (this is the path the shipped default
config and the real full training run both take — `torch.utils.data.Subset`
is never even constructed).

---

## 4. `src/train.py` — the training loop

### 4a. Config resolution

```python
DEFAULT_CONFIG_SEARCH_PATH = [
    "/app/configs/training_config.yaml",
    str(Path(__file__).resolve().parent.parent / "configs" / "training_config.yaml"),
]

def resolve_config_path() -> str:
    env_path = os.environ.get("TRAINING_CONFIG_PATH")
    if env_path:
        return env_path
    for candidate in DEFAULT_CONFIG_SEARCH_PATH:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(...)
```

Three-tier resolution: explicit env var first (what both the Docker image
and the Kubernetes Job set), then the conventional container mount path,
then a local dev-relative path (so you can run `python src/train.py`
straight from a git checkout with no Docker/env-var setup at all). This
is what lets the *same* `train.py` run unmodified in three different
contexts — bare `python`, `docker run` with a mounted config, and a
Kubernetes `Job` with a `ConfigMap` mounted volume — because each context
just needs to satisfy one of the three lookup rules.

### 4b. The training/eval loop

```python
for epoch in range(config["training"]["epochs"]):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    log_entry = {"event": "epoch_complete", "epoch": epoch + 1, ...}
    print(json.dumps(log_entry), flush=True)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({...}, save_path)
        print(json.dumps({"event": "checkpoint_saved", ...}), flush=True)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(json.dumps({"event": "early_stopping", ...}), flush=True)
            break
```

Standard supervised training loop: `Adam` optimizer, `CrossEntropyLoss`
(the correct loss for multi-class single-label classification with raw
logit outputs — it internally applies `log_softmax`, so the model's `fc`
layer correctly outputs raw logits, not probabilities). Each epoch trains
on the full train split, then evaluates on the held-out validation
(test) split.

**Checkpointing strategy: save only when validation loss improves.** This
is a deliberate choice over "save every epoch" or "save the last epoch" —
it means `classifier_v1.pt` on disk is always the *best* model seen
during the run, never an epoch that started overfitting afterward. The
checkpoint dict stores the architecture name, class count, and both loss
and accuracy alongside the weights — this is what makes it
**self-describing** (see §2's factory-function discussion): `serve.py`
never needs to be told separately what architecture to build.

**Early stopping**: if validation loss hasn't improved for
`early_stopping_patience` (3) consecutive epochs, training stops early.
This guards against wasting compute (or letting a long-running Job run
needlessly) once the model has stopped improving. Note the loop iterates
`range(config["training"]["epochs"])` — the max epoch count — but early
stopping is a `break`, so a run can finish faster than the configured
max, but never slower.

### 4c. JSON-line structured logging

Every event (`config_loaded`, `device_selected`, `epoch_complete`,
`checkpoint_saved`, `early_stopping`/`training_complete`) is printed as a
single-line JSON object, `flush=True`. This is deliberate structured
logging, not human-readable prose:
- Every line is independently machine-parseable (`json.loads(line)`) —
  useful for feeding logs into a log aggregator, or for a script to
  `grep`/parse metrics out of `kubectl logs` output without regex.
- `flush=True` matters specifically in containers: stdout is normally
  line-buffered when attached to a terminal but **block-buffered** when
  piped (as it always is under Docker/Kubernetes' log capture) — without
  explicit flushing, you could watch `docker logs -f`/`kubectl logs -f`
  and see nothing for a long time even though the process is actively
  producing output, because it's stuck in an internal buffer.

---

## 5. `src/serve.py` — the inference API

### 5a. Startup: loading the model exactly once

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = load_model()
    yield

app = FastAPI(title="MLOps PyTorch CIFAR-10 Classifier", lifespan=lifespan)
```

FastAPI's `lifespan` context manager runs its code before the app starts
accepting requests, and again (the code after `yield`, here empty) on
shutdown. Loading the model here — once, at process startup — rather than
per-request means every `/predict` call reuses the same in-memory model;
re-loading a ~130MB checkpoint from disk on every request would be both
slow and wasteful. `_model` is a module-level global specifically so
`/health` and `/predict` (defined as separate functions) can both read it.

### 5b. `/health` — used by Kubernetes probes

```python
@app.get("/health")
def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}
```

Returns HTTP 200 only if the model actually loaded successfully; 503
otherwise. This one endpoint backs **both** the Deployment's
`livenessProbe` and `readinessProbe` (see §7) — the same check answers two
different Kubernetes questions ("is this container alive at all" vs. "is
this container ready to receive traffic"), which is valid here because
for this service there's no meaningful difference between "alive" and
"ready" — a serving pod with no model loaded is neither.

### 5c. `/predict` — inference

```python
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, ...)
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    tensor = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).tolist()

    predicted_idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
    return {"predicted_class": CIFAR10_CLASSES[predicted_idx], "probabilities": {...}}
```

Accepts a `multipart/form-data` file upload (not raw bytes or base64 —
`UploadFile`/`File(...)` is FastAPI's idiomatic way to accept binary file
uploads, and matches what `curl -F "image=@file.png"` sends). Any image
format Pillow can decode works, converted to RGB (so a grayscale or
paletted/RGBA PNG doesn't break the 3-channel model input). Errors during
image decoding are caught and turned into a `400 Bad Request` with the
underlying reason — a malformed upload is a *client* error, not a server
fault, so it must not be a 500.

`torch.no_grad()` disables autograd bookkeeping — inference never needs
gradients, so this saves memory and compute. `F.softmax` converts raw
logits into a proper probability distribution over the 10 classes
(non-negative, sums to 1) before returning it — raw logits are relative
scores useful for training but not directly interpretable as
"confidence" the way softmax output is. The response includes **both**
the single predicted class and the full probability distribution over all
10 classes, rather than just the top prediction — useful for a caller
that wants to inspect confidence/runner-up classes, not just the argmax.

---

## 6. Docker — `docker/Dockerfile.train` and `docker/Dockerfile.serve`

Both are **multi-stage builds** with the same two-stage shape: a `base`
stage that installs Python dependencies, then a second stage that copies
in only the application code needed for that specific role.

```dockerfile
# Dockerfile.train
FROM python:3.11-slim AS base
WORKDIR /app
COPY requirements/train.txt .
RUN pip install --no-cache-dir -r train.txt

FROM base AS training
COPY src/model.py src/dataset.py src/train.py ./src/
COPY configs/ ./configs/
ENV PYTHONUNBUFFERED=1
ENV TRAINING_CONFIG_PATH=/app/configs/training_config.yaml
ENTRYPOINT ["python", "src/train.py"]
```

```dockerfile
# Dockerfile.serve
FROM python:3.11-slim AS base
WORKDIR /app
COPY requirements/serve.txt .
RUN pip install --no-cache-dir -r serve.txt

FROM base AS serving
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY src/model.py src/dataset.py src/serve.py ./src/
RUN groupadd --gid 1000 appuser && useradd --uid 1000 --gid appuser --shell /bin/false --create-home appuser \
    && mkdir -p /app/checkpoints && chown -R appuser:appuser /app
USER appuser
ENV PYTHONUNBUFFERED=1
ENV CHECKPOINT_PATH=/app/checkpoints/classifier_v1.pt
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
ENTRYPOINT ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8080"]
```

Why each choice matters:

- **`python:3.11-slim` base**, not the full `python:3.11` image: the
  slim variant strips out build toolchains, docs, and other non-runtime
  packages, cutting image size substantially with no loss of Python
  functionality that this app needs.
- **`--no-cache-dir` on pip installs**: pip normally caches downloaded
  wheels under `~/.cache/pip` for reuse across installs; inside a
  container that cache is dead weight that only bloats the final image
  since there will never be a second `pip install` in the same layer.
- **`requirements/*.txt` copied and installed *before* `src/` is
  copied in**: Docker layer caching means changing application code
  (which happens often) doesn't invalidate the (slow) dependency-install
  layer, as long as the requirements files themselves haven't changed.
  Reordering this (copying all source first, then installing deps) would
  mean every code change re-triggers a full `pip install`.
- **Only the files each role actually needs are `COPY`'d**: the training
  image never gets `serve.py`, the serving image never gets `train.py` or
  `configs/`. This isn't just tidiness — it minimizes the attack surface
  and image size, and it also means a `docker run` of the serving image
  literally *cannot* accidentally kick off a training run even if
  misconfigured, because the code to do so isn't present in that image.
- **Non-root user in the serving image (`appuser`, uid 1000)**: running
  containers as root is a well-known security anti-pattern — if the
  process is ever compromised (e.g. via a dependency vulnerability), a
  root process inside the container has a larger blast radius (e.g.
  easier container-escape primitives) than a non-privileged one. The
  training image does *not* drop to a non-root user — it's a short-lived
  batch job with no exposed network port and no long-running server
  process, so the risk profile is different, and this is a defensible,
  intentional asymmetry rather than an oversight.
- **`HEALTHCHECK` in the serving image only**: Docker's built-in
  healthcheck mechanism polls `curl -f http://localhost:8080/health`
  every 10s (after a 15s startup grace period, 3s timeout, 3 retries
  before marking unhealthy) — this makes `docker ps` itself show
  `(healthy)`/`(unhealthy)` for the container, independent of Kubernetes.
  The training image has no long-running process to health-check — it
  runs to completion and exits, so a `HEALTHCHECK` wouldn't have anything
  meaningful to poll.
- **`ENV PYTHONUNBUFFERED=1`**: the container-level version of the same
  problem `flush=True` solves in Python code (§4c) — this environment
  variable disables Python's own stdout buffering globally, so `print()`
  output (and anything from `uvicorn`) reaches `docker logs`/`kubectl
  logs` immediately rather than sitting in a buffer.
- **Port 8080, not 80**: binding to ports below 1024 traditionally
  requires root privileges on Linux; since the serving container runs as
  a non-root user, it must listen on an unprivileged port. 8080 is the
  conventional choice for this. The Kubernetes `Service` (§7) is what
  actually exposes this on the conventional port 80 to *callers* of the
  service, translating `80 → 8080` — callers never need to know the pod
  listens on 8080 internally.
- **`--extra-index-url https://download.pytorch.org/whl/cpu` in both
  requirements files**: PyPI's default `torch` wheel bundles the full
  CUDA runtime (multiple GB) even on machines with no GPU. Pointing at
  PyTorch's dedicated CPU-only wheel index cuts the training image from
  5.1GB down to 1.1GB with zero behavior change on CPU-only hosts —
  `torch.cuda.is_available()` still correctly returns `False`.

### `.dockerignore`

Excludes `.git/`, `.github/`, `data/`, `checkpoints/`, virtualenvs,
`__pycache__/`, `tests/`, `k8s/`, and `*.md` from the Docker build
context. Two purposes: (1) keeps the ~170MB downloaded CIFAR-10 tarball
and multi-hundred-MB checkpoints from ever being sent to the Docker
daemon as build context (slow, and would risk accidentally baking a
dataset into an image layer); (2) keeps genuinely irrelevant files
(K8s manifests, tests, markdown docs) out of the build context entirely,
since neither Dockerfile's `COPY` instructions reference them anyway.

---

## 7. Kubernetes manifests (`k8s/`)

All resources live in a dedicated `ml-training` `Namespace`
(`namespace.yaml`) — namespacing everything under one project-specific
name keeps this workload's resources cleanly separable from anything else
that might run on the same cluster, and lets you `kubectl delete
namespace ml-training` to tear down everything in one shot.

### 7a. `configmap.yaml` — externalized hyperparameters

A `ConfigMap` named `training-config` holds a literal copy of
`training_config.yaml`'s contents as data. It's mounted into the training
Job as a **read-only volume** at `/app/configs`, which the container then
reads via the same `TRAINING_CONFIG_PATH` env var mechanism used
everywhere else (§4a). This is the standard Kubernetes pattern for
externalizing configuration from images: changing hyperparameters means
editing/reapplying the `ConfigMap` and re-running the `Job`, not rebuilding
the training image.

### 7b. `training-job.yaml` — the training workload

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: cifar10-training-job
  namespace: ml-training
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: trainer
          image: mlops-train:v1
          env:
            - name: TRAINING_CONFIG_PATH
              value: /app/configs/training_config.yaml
            - name: OMP_NUM_THREADS
              value: "2"
            - name: MKL_NUM_THREADS
              value: "2"
          volumeMounts:
            - {name: training-config, mountPath: /app/configs, readOnly: true}
            - {name: training-data, mountPath: /app/data}
            - {name: model-checkpoints, mountPath: /app/checkpoints}
          resources:
            requests: {cpu: "2", memory: 4Gi}
            limits: {cpu: "2", memory: 4Gi}
      volumes:
        - {name: training-config, configMap: {name: training-config}}
        - {name: training-data, persistentVolumeClaim: {claimName: training-data-pvc}}
        - {name: model-checkpoints, persistentVolumeClaim: {claimName: model-checkpoints-pvc}}
```

Plus two `PersistentVolumeClaim`s defined in the same file:
`training-data-pvc` (2Gi, `ReadWriteOnce`) for the CIFAR-10 dataset cache,
`model-checkpoints-pvc` (1Gi, `ReadWriteOnce`) for the output checkpoint.

**Why a `Job`, not a `Deployment` or bare `Pod`:** training is a
**run-to-completion** task, not a long-running service. A `Deployment`
exists to keep N replicas *continuously* running (restarting them
forever if they exit) — completely wrong semantics for something that's
supposed to finish and stop. A `Job` specifically models "run this to
completion, track success/failure, don't just restart it forever."
`restartPolicy: Never` plus `backoffLimit: 1` means: if the training
process crashes, retry once, then give up and mark the Job failed rather
than retrying indefinitely — appropriate for a workload where a crash is
likely a real bug, not a transient blip worth infinite retries.

**Why two separate PVCs instead of one:** they have different lifecycles
and different consumers. `training-data-pvc` is written once by the
training Job and never needed again after training. `model-checkpoints-pvc`
is written by the training Job but then **read by the serving Deployment**
(mounted read-only there, see §7c) — it's the hand-off artifact between
the training and serving halves of the pipeline. Splitting them means the
serving Deployment's volume mount list doesn't need to reference the
(irrelevant to serving) training-data volume at all.

**`resources.requests` == `resources.limits`** (both `cpu: "2"`, both
`memory: 4Gi`): this is a **Guaranteed** QoS class pod in Kubernetes
terms — requests equal to limits means Kubernetes won't ever throttle or
OOM-kill this pod for "misbehaving" relative to its own request, and it
gets the highest eviction priority protection under node pressure. For a
one-shot training job you want predictable, complete resource allocation,
not the possibility of being squeezed mid-run.

**The `OMP_NUM_THREADS`/`MKL_NUM_THREADS` env vars** — this is the fix for
the CPU-throttling bug discovered during real validation (full story in
§9). The short version: PyTorch sizes its intra-op thread pool to the
*host's* visible CPU count, not the pod's cgroup CPU quota, so without
these it will spin up far more threads than the pod's `cpu: "2"` limit
can actually run concurrently, and nearly all of them spend their time
being throttled by the Linux CFS scheduler rather than doing useful work.

### 7c. `serving-deployment.yaml` — the serving workload

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate: {maxSurge: 1, maxUnavailable: 0}
  template:
    spec:
      containers:
        - name: serving
          image: mlops-serve:v1
          ports: [{containerPort: 8080}]
          volumeMounts:
            - {name: model-checkpoints, mountPath: /app/checkpoints, readOnly: true}
          livenessProbe: {httpGet: {path: /health, port: 8080}, periodSeconds: 10, failureThreshold: 3}
          readinessProbe: {httpGet: {path: /health, port: 8080}, periodSeconds: 5, initialDelaySeconds: 15}
          resources:
            requests: {cpu: 500m, memory: 1Gi}
            limits: {cpu: "1", memory: 2Gi}
      volumes:
        - {name: model-checkpoints, persistentVolumeClaim: {claimName: model-checkpoints-pvc, readOnly: true}}
```

**Why a `Deployment`, not a `Job`:** the inverse of §7b's reasoning —
serving is a long-running service that should keep running (and be
restarted/rescheduled if a pod dies), which is exactly `Deployment`
semantics.

**2 replicas by default**: baseline redundancy — if one pod is
unhealthy, restarting, or its node fails, the other keeps serving traffic.
This is also the `HorizontalPodAutoscaler`'s `minReplicas` (§7e).

**`RollingUpdate` with `maxSurge: 1, maxUnavailable: 0`**: when the
Deployment is updated (e.g. a new image version), Kubernetes may create
up to 1 *extra* pod above the desired count (`maxSurge: 1`) but must never
drop below the full desired count of *available* pods (`maxUnavailable:
0`) during the rollout. In practice, with 2 replicas: it creates a 3rd
pod running the new version, waits for it to pass its readiness probe,
*then* terminates one old pod, and repeats — meaning capacity for serving
traffic never dips below 2 ready pods at any point during a rollout. This
is the safest (if slightly slower/more resource-hungry) rollout strategy
available.

**`livenessProbe` vs `readinessProbe` — both hit `/health`, but they mean
different things to Kubernetes:**
- **Liveness** (`periodSeconds: 10, failureThreshold: 3` → the pod is
  killed and restarted if `/health` fails 3 consecutive checks, i.e.
  after ~30s of continuous failure): answers "is this container in a
  broken state that only a restart can fix?"
- **Readiness** (`periodSeconds: 5, initialDelaySeconds: 15` → checked
  every 5s, but not started until 15s after container start): answers
  "should the Service currently route traffic to this pod?" A pod that
  fails readiness is pulled out of the Service's load-balancing rotation
  *without* being killed/restarted — it stays running so it has a chance
  to recover, but stops receiving new requests in the meantime.
- **Why readiness needs `initialDelaySeconds: 15` and liveness doesn't**:
  the container needs time to load the model checkpoint at startup
  (`lifespan`, §5a) before `/health` will ever return 200. Without an
  initial delay, the *readiness* probe would immediately mark the pod
  not-ready (fine — that's the correct state during startup) — but
  without a startup grace mechanism, an aggressive *liveness* probe could
  in principle kill a pod that's still legitimately loading. Here
  liveness compensates via its own `failureThreshold: 3` × `periodSeconds:
  10` = 30s grace window instead of an explicit delay — both probes
  effectively tolerate the same kind of startup lag, just through
  different mechanisms.

**Checkpoints volume mounted `readOnly: true` in the Deployment** (versus
read-write in the training Job): the serving Deployment must never be
able to modify or delete the checkpoint it's serving — it only consumes
the artifact the training Job produced. This is a real safety boundary,
not just documentation-by-comment: a bug in `serve.py` that tried to write
to that path would fail at the filesystem/mount level, not just by
convention.

**`resources.requests` (500m/1Gi) below `resources.limits` (1/2Gi)** —
unlike the training Job's Guaranteed QoS, this is intentionally
**Burstable** QoS: a serving pod's actual load varies with traffic, so it
should be able to burst above its baseline request when handling a spike,
while still guaranteeing a request-level floor for the scheduler to
reason about when placing pods on nodes.

### 7d. `serving-service.yaml` — stable network identity

```yaml
apiVersion: v1
kind: Service
spec:
  type: ClusterIP
  selector: {app: model-serving}
  ports: [{name: http, port: 80, targetPort: 8080}]
```

Pods are ephemeral — they get new IPs whenever they're rescheduled. A
`Service` provides a stable virtual IP/DNS name (`model-serving`, reachable
cluster-internally at `model-serving.ml-training.svc.cluster.local`) that
load-balances across every pod currently matching `selector: {app:
model-serving}` — i.e. every ready pod from the Deployment above, since
the Deployment's pod template carries that exact label. `port: 80 →
targetPort: 8080` is the translation mentioned in §6: cluster-internal
callers hit the conventional port 80, and the Service forwards to
whatever unprivileged port the pods actually listen on. `type: ClusterIP`
(rather than `NodePort`/`LoadBalancer`) means this is reachable only from
*inside* the cluster by default — accessing it from outside (as done
during validation) goes through `kubectl port-forward`, which is
appropriate for a course assignment/local-dev context; a production
deployment behind a real ingress or cloud load balancer would swap in a
different Service type or add an `Ingress` on top, without changing
anything else in this manifest set.

### 7e. `hpa.yaml` — autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef: {apiVersion: apps/v1, kind: Deployment, name: model-serving}
  minReplicas: 2
  maxReplicas: 5
  metrics: [{type: Resource, resource: {name: cpu, target: {type: Utilization, averageUtilization: 70}}}]
```

Watches the average CPU utilization (as a percentage of each pod's
*requested* CPU, i.e. against the 500m request from §7c, not the 1-core
limit) across all pods in the `model-serving` Deployment. If average
utilization exceeds 70%, it scales the Deployment up (up to 5 replicas
max); if it drops well below 70% sustained, it scales back down (never
below 2, the same floor as the Deployment's own baseline replica count —
so the HPA can only ever move replica count *up* from the deployed
baseline, or back down to it). This requires the `metrics-server`
component to be running in the cluster to supply actual CPU utilization
numbers to the HPA controller — a standard cluster add-on, and part of
what was installed/verified as working during the `kind` cluster
validation.

### 7f. `training-job-gpu.yaml` — the bonus GPU variant

Structurally identical to `training-job.yaml` (same PVCs, same
ConfigMap/volume wiring, same thread-pinning env vars) with three
additions: a `nvidia.com/gpu: 1` entry in both `resources.requests` and
`resources.limits`, a `nodeSelector: {accelerator: nvidia-gpu}` to
constrain scheduling to GPU-labeled nodes, and a `toleration` for the
`nvidia.com/gpu=present:NoSchedule` taint that GPU node pools are
conventionally tainted with (so that non-GPU workloads don't
accidentally get scheduled onto expensive GPU nodes — this toleration is
what lets *this* Job specifically bypass that taint). This variant was
not applied during the `kind`-based validation (`kind` nodes have no
GPUs to expose), but demonstrates the adaptation a real GPU cluster would
need — same training code, same image, only the K8s scheduling directives
change. Note `train.py`/`model.py` already handle this automatically —
`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` —
no code change is needed to actually use a GPU if one is present, only
the K8s-level request for one.

---

## 8. CI/CD — `.github/workflows/ci.yml`

```yaml
on:
  push: {branches: [main, develop]}
  pull_request: {branches: [main, develop]}

jobs:
  lint-and-test:
    steps: [checkout, setup-python 3.11, pip install deps + pytest + ruff,
             "ruff check src/ tests/", "pytest tests/ -v"]
  docker-build:
    needs: lint-and-test
    steps: [checkout, setup buildx, build Dockerfile.train, build Dockerfile.serve]
```

Two sequential jobs, the second gated on the first (`needs:
lint-and-test`) — no point spending CI minutes building Docker images if
the code doesn't even lint or pass its unit tests. Triggers on push *and*
pull_request against both `main` and `develop`, so every PR gets checked
before merge, and direct pushes to either protected-ish branch are also
verified.

`docker-build` doesn't `push` the images anywhere (`push: false`) — its
purpose here is purely to prove the Dockerfiles still build successfully
as source changes land, using GitHub Actions' layer cache
(`cache-from/cache-to: type=gha`) to keep repeated builds fast. There's no
container registry wired up in this assignment's scope, so there's
nothing to push to.

### `ruff.toml`

```toml
line-length = 100
[lint]
select = ["E", "F", "W"]
```

Ruff's *default* rule set (and especially its "preview" rules) is large
and opinionated well beyond basic style — running with no config produced
false-positive-feeling failures for things like preferring `from torch
import nn` over `import torch.nn as nn`, or flagging FastAPI's idiomatic
`File(...)` default-argument pattern as risky. Pinning `select` to just
`E`/`F`/`W` (pycodestyle errors/warnings + pyflakes) keeps CI enforcing
genuinely useful checks (unused imports, undefined names, line length,
actual syntax-adjacent issues) without fighting the codebase's existing,
intentional conventions.

---

## 9. Bugs found and fixed during real validation

Both of these were discovered by actually *running* the pipeline for
real (not just writing manifests) — they're the kind of failure that only
shows up once code leaves a "happy path" dev environment, which is
exactly the point of Part F's end-to-end validation requirement.

### 9a. MKL-DNN SIGFPE on AMD EPYC

**Symptom**: `Conv2d` forward passes crashed with `SIGFPE` — process exit
code 136, no Python traceback at all (a native-code crash below the
Python interpreter). Plain tensor arithmetic was fine; only actual
convolution ops crashed.

**How it was isolated**: stepped through ResNet-18 layer-by-layer with
flushed prints until the crash localized to the very first conv layer,
then reproduced it with a bare `nn.Conv2d` on a synthetic tensor (no
model, no dataset) to rule out anything CIFAR/ResNet-specific.

**What didn't work**: several documented workarounds for this general
class of bug (`MKL_ENABLE_INSTRUCTIONS`, `MKL_DEBUG_CPU_TYPE`,
`ATEN_CPU_CAPABILITY` environment variables) had no effect.

**Root cause and fix**: a known oneDNN CPU-instruction-dispatch bug
affecting some AMD EPYC hosts, inside PyTorch's MKL-DNN backend
specifically. Disabling that backend entirely
(`torch.backends.mkldnn.enabled = False`, at the top of `model.py`, §2c)
fixed it — convolutions route through a different, correct, slightly
slower code path instead.

### 9b. PyTorch thread pool vs. Kubernetes CPU quota (discovered during Part F)

**Symptom**: the training `Job` on the `kind` cluster appeared to hang
for 20+ minutes with zero progress, despite the exact same image/config
running fine under plain `docker run` on the same machine.

**How it was diagnosed**: `kubectl exec ... cat /sys/fs/cgroup/cpu.stat`
inside the pod showed `nr_throttled` at ~99% of `nr_periods` — i.e. the
kernel's CFS scheduler was throttling this pod's CPU access almost every
single scheduling period. `kubectl exec ... python -c "import torch;
print(torch.get_num_threads())"` returned **16** — the *host* machine's
full core count — while the pod's `resources.limits.cpu` was `"2"`.

**Root cause**: PyTorch sizes its intra-op thread pool using
`os.cpu_count()` by default, which reflects CPUs visible on the
underlying *host*, completely blind to the pod's Kubernetes CFS quota.
16 threads were contending for scheduler time that a 2-core cgroup quota
could only actually grant to 2 of them at once — the rest spent nearly
all their time blocked/throttled rather than computing.

**Fix**: `OMP_NUM_THREADS: "2"` and `MKL_NUM_THREADS: "2"` env vars,
added to the Job container spec (§7b), pinning PyTorch's thread pool to
match the CPU request/limit. Verified: `torch.get_num_threads()` then
correctly returned 2, throttling dropped from ~99% to ~33%
(`nr_periods`/`nr_throttled` ratio), and the same Job that never completed
in 20+ minutes before the fix completed in under 3 minutes after it.

**Why this matters beyond this one bug**: it's a general lesson about
CPU-bound native/multi-threaded workloads (not just PyTorch — any
OpenMP/MKL/BLAS-backed library has the same class of risk) running under
Kubernetes: **setting `resources.limits.cpu` alone does not tell the
*process itself* to size its own concurrency accordingly.** The cgroup
quota is enforced by the kernel scheduler *after* the fact (by
throttling), not communicated to the application beforehand. Any
thread-pool-sizing library defaulting to host CPU count needs an explicit
override in containerized environments with CPU limits below the host's
core count.

---

## 10. The git/PR workflow actually used

```
main
 └── develop
      ├── feature/repo-scaffold      (PR #1 → develop)  — structure, .gitignore, CI skeleton
      ├── feature/pytorch-model      (PR #2 → develop)  — model/dataset/train/serve
      ├── feature/docker-training    (PR #3 → develop)  — Dockerfiles, verified locally
      └── feature/k8s-deployment     (PR #4 → develop)  — K8s manifests, validated on kind
 develop → main                      (PR #5, release)
 docs/final-pr-links → main          (PR #6, follow-up docs fix)
```

Each feature branch corresponds to one part of the assignment (roughly
Parts A/B/C/D+E+F) and was merged via its own PR with a [Conventional
Commits](https://www.conventionalcommits.org/)-style message (`feat: ...`,
`chore: ...`). `develop` accumulated all four feature merges, then one
release PR (`develop → main`) promoted the whole accumulated set to
`main` at once — a common pattern for keeping `main` limited to
release-quality snapshots while `develop` is where ongoing feature work
lands and integrates first. `main`/`develop` were kept in sync afterward
via fast-forward merges whenever one advanced without the other (e.g. PR
#6 landed directly against `main` for a small doc fix, then `develop` was
fast-forwarded to match).

This satisfies the assignment's structural requirements: ≥4 merged PRs (6
were actually merged), a `main`/`develop`/feature-branch structure, and
Conventional Commits messages throughout.
