# Study guide: likely questions and model answers

A self-test companion to [`WALKTHROUGH.md`](WALKTHROUGH.md). Questions are
grouped by topic and roughly ordered from "what did you build" to "why did
you build it this way" to "what would happen if...". Use this to rehearse
explaining the project out loud — that's a different skill than reading
the code, and it's usually what gets tested.

---

## A. Big picture / architecture

**Q: In one or two sentences, what does this project do?**
> A: Trains a ResNet-18 image classifier on CIFAR-10 with PyTorch, serves
> it behind a FastAPI `/predict` endpoint, packages both as Docker images,
> and deploys both to Kubernetes — a training `Job` that produces a
> checkpoint, and a serving `Deployment`/`Service`/`HPA` that consumes it.

**Q: Why is training a Kubernetes `Job` and serving a `Deployment`, not
both the same kind of resource?**
> A: A `Job` models run-to-completion work — it starts, does the task
> once, exits, and Kubernetes tracks success/failure rather than
> restarting it forever. A `Deployment` models a long-running service
> that should always have N replicas up, restarting them if they die.
> Training fits the first model; serving fits the second. Using a
> `Deployment` for training would mean Kubernetes keeps relaunching a
> finished training process forever, which is wrong.

**Q: How does the checkpoint get from the training side to the serving
side?**
> A: Both the training `Job` and the serving `Deployment` mount the same
> `model-checkpoints-pvc` `PersistentVolumeClaim` — the Job mounts it
> read-write and writes `classifier_v1.pt` there; the Deployment mounts
> the *same* PVC read-only and loads that file at startup. It's the
> hand-off artifact between the two halves of the pipeline; there's no
> registry or explicit "promote model" step in this design.

**Q: What would you have to change to run this on a real cloud cluster
instead of `kind`?**
> A: Push the two images to a real registry (ECR/GCR/Docker Hub) instead
> of `kind load docker-image`, and update the image references
> accordingly. The PVCs would need a `StorageClass` the cloud provider
> actually backs (kind uses `local-path-provisioner`, which doesn't exist
> outside kind). The `ClusterIP` Service would likely need an `Ingress` or
> a `LoadBalancer`-type Service in front of it to be reachable from
> outside the cluster. Everything else — the Job, Deployment, ConfigMap,
> HPA — is portable as-is; that portability was the point of using
> standard manifests rather than kind-specific tooling.

---

## B. The model (`src/model.py`)

**Q: Why ResNet-18, and why not the stock `torchvision` version?**
> A: ResNet-18 is a well-understood, well-tested architecture — the point
> of this assignment is the infrastructure around training/serving/
> deploying, not novel modeling, so a standard architecture keeps the
> focus there. The *stock* version assumes 224×224 ImageNet images: its
> stem (7×7 stride-2 conv + stride-2 maxpool) downsamples by 4× before any
> residual block runs, which would crush a 32×32 CIFAR image down to 8×8
> almost immediately. The adapted version swaps in a 3×3 stride-1 stem
> conv and removes the maxpool (`nn.Identity()`), which is the standard
> CIFAR-ResNet adaptation.

**Q: Why is `pretrained` `False` by default?**
> A: The stem conv is being replaced regardless, so ImageNet-pretrained
> weights for *that* layer would be thrown away anyway. Using
> ImageNet-pretrained weights for the rest of the network while training
> on 32×32 CIFAR images is a legitimate transfer-learning option, but not
> one this assignment calls for, so the default is training from scratch,
> with `pretrained=True` available as an opt-in.

**Q: What's `SimpleCNN` for, and is it actually used anywhere?**
> A: It's a smaller, from-scratch 3-conv-block CNN offered as a
> lighter/faster alternative architecture, selectable via the same
> `get_model(architecture=...)` factory. It's not used by the shipped
> default config (which specifies `resnet18`), but it is exercised by the
> parametrized unit tests, proving the architecture-swapping mechanism
> genuinely works end-to-end, not just for one hardcoded model.

**Q: Explain `torch.backends.mkldnn.enabled = False` at the top of
`model.py`. Why is it there, and why at that location specifically?**
> A: It works around a real bug found during this project: on this
> machine's AMD EPYC CPU, PyTorch's MKL-DNN (oneDNN) backend crashes
> `Conv2d` forward passes with `SIGFPE` due to a CPU-instruction-dispatch
> bug. Disabling MKL-DNN routes convolutions through a different, correct
> (if slightly slower) code path. It's placed at module import time in
> `model.py`, not in `train.py`/`serve.py` individually, so that *every*
> entry point that imports the model gets the fix automatically — there's
> no way to construct a model without it taking effect first.

**Q: How does `serve.py` know which architecture to build when loading a
checkpoint?**
> A: The checkpoint dict itself stores `architecture` and `num_classes`
> alongside the weights (`torch.save({"architecture": ..., "model_state_dict":
> ..., ...})` in `train.py`). `serve.py`'s `load_model()` reads those back
> out of the checkpoint and passes them into the same `get_model()`
> factory. This makes checkpoints self-describing — serving never needs a
> separately-maintained "what architecture is this" config.

---

## C. Data (`src/dataset.py`)

**Q: Why do training and evaluation use different transforms?**
> A: Training applies data augmentation (`RandomHorizontalFlip`,
> `RandomCrop(32, padding=4)`) to create synthetic variation and reduce
> overfitting/memorization. Evaluation and inference use no augmentation
> — you want a deterministic transform of the actual input when measuring
> accuracy or serving a real prediction, not a randomly perturbed one.
> Both share identical normalization constants, because normalization
> *must* match between train and inference or the model sees a
> distribution shift it was never trained on.

**Q: Why does `serve.py` reuse `get_transforms(train=False)` from
`dataset.py` instead of writing its own preprocessing?**
> A: So inference preprocessing is guaranteed identical to how validation
> images were preprocessed during training — there's a single source of
> truth for "how do you turn an image into a model input tensor," and it
> can't silently drift out of sync between the two files.

**Q: What is `subset_fraction` and why does it exist?**
> A: A knob that deterministically shrinks the dataset to a fraction of
> its size (fixed seed 42, so it's the same subsample every run). It
> exists purely to make local/CI verification runs fast — the shipped
> production config uses `subset_fraction: 1.0` (the full dataset,
> effectively a no-op), while a separate `training_config.verify.yaml`
> uses `0.05` for quick pipeline smoke-tests.

**Q: If `subset_fraction` is `1.0`, does the code still go through the
subsetting logic?**
> A: No — `_maybe_subset()` checks `if fraction >= 1.0: return dataset`
> and returns the original dataset untouched. `torch.utils.data.Subset`
> is never constructed on that path, so the full run has zero overhead
> from this feature.

**Q: Why `download=True` on the `CIFAR10` dataset constructor — doesn't
that mean it re-downloads every run?**
> A: No — torchvision's `download=True` means "download *if not already
> present* at `root`," and it verifies via checksums it already has locally.
> The first run against an empty `data_dir` downloads and extracts the
> ~170MB tarball; every subsequent run against the same mounted
> directory/PVC finds the files already there and skips the network call
> entirely. This is exactly why volume-mounting `/app/data` matters for
> Docker/Kubernetes: the cache persists across container runs.

---

## D. Training (`src/train.py`)

**Q: Walk me through config resolution — how does `train.py` know where
its YAML config is?**
> A: Three tiers, checked in order: (1) the `TRAINING_CONFIG_PATH`
> environment variable, if set — this is what both the Docker image and
> the Kubernetes Job set explicitly; (2) the conventional mounted path
> `/app/configs/training_config.yaml`; (3) a path relative to the script
> location, for running straight from a git checkout with no env var or
> mount at all. This lets the identical script run unmodified across bare
> `python`, `docker run`, and Kubernetes contexts.

**Q: Why `CrossEntropyLoss`, and does the model's output need to be
softmax'd before computing it?**
> A: `CrossEntropyLoss` is the correct loss for multi-class,
> single-label classification. It internally applies `log_softmax` to its
> input, so the model must output raw logits (which it does — `fc`'s
> output is unmodified) — applying softmax before passing to
> `CrossEntropyLoss` would be double-softmaxing and wrong.

**Q: What's the checkpointing policy — save every epoch, save the last
epoch, or something else?**
> A: Save only when validation loss improves over the best seen so far.
> This guarantees the checkpoint on disk is always the best model
> observed during the run, never a later epoch that may have started
> overfitting.

**Q: Explain early stopping here — what's `patience`, and what happens
when it's exceeded?**
> A: `early_stopping_patience` (3 in the shipped config) counts
> *consecutive* epochs with no validation-loss improvement. When that
> counter reaches the patience value, training breaks out of the epoch
> loop immediately rather than continuing to the configured max
> (`epochs: 10`). It's a `break`, so a run can finish in fewer epochs than
> configured, never more.

**Q: Why JSON-lines logging instead of normal human-readable print
statements?**
> A: Every line is independently machine-parseable
> (`json.loads(line)`) — useful for piping `kubectl logs`/`docker logs`
> output into a log aggregator or a script that extracts metrics, without
> regex-scraping prose. It's also just as readable to a human scanning the
> raw output, so nothing is lost either way.

**Q: Why does every print call pass `flush=True`?**
> A: stdout is block-buffered (not line-buffered) when it's piped rather
> than attached to an interactive terminal — which is always the case
> under Docker/Kubernetes log capture. Without `flush=True`, output could
> sit in an internal buffer for a long time even while the process is
> actively working, making `docker logs -f`/`kubectl logs -f` appear to
> hang. `ENV PYTHONUNBUFFERED=1` in the Dockerfiles is the same fix
> applied at the container level, covering any output this explicit
> `flush=True` doesn't (e.g. uvicorn's own logs in the serving image).

---

## E. Serving (`src/serve.py`)

**Q: Why load the model in a `lifespan` context manager instead of at
request time inside `/predict`?**
> A: Loading a ~130MB checkpoint from disk is slow; doing it once at
> process startup means every request reuses the same in-memory model
> instead of reloading it per-request, which would be both slow and
> wasteful. `lifespan` is FastAPI's mechanism for startup/shutdown code
> that runs exactly once around the app's request-serving lifetime.

**Q: `/health` backs both the liveness and readiness probes in
Kubernetes. Is that a problem?**
> A: Not here — liveness asks "is this container broken enough to need a
> restart," readiness asks "should traffic currently be routed to this
> pod." For this service, a pod with no model loaded is neither alive in
> a useful sense nor ready for traffic, so answering both questions with
> the same "did the model load" check is a legitimate simplification, not
> a shortcut that loses information. A more complex service with
> independent liveness/readiness semantics (e.g. "alive but temporarily
> overloaded") would need separate endpoints.

**Q: What happens if `/predict` is called with a corrupted or non-image
file?**
> A: `Image.open()` raises inside the `try` block, which is caught and
> re-raised as `HTTPException(status_code=400, detail=f"Invalid image:
> {exc}")`. It's a 400, not a 500 — a bad upload is a client error, and
> the server correctly handled it as an expected failure case rather than
> crashing.

**Q: Why does `/predict` return the full probability distribution over
all 10 classes, not just the top prediction?**
> A: The predicted class alone loses information a caller might want —
> e.g. how confident the model was, or what the runner-up classes were.
> Returning both `predicted_class` (the argmax) and the full
> `probabilities` dict costs almost nothing extra and is strictly more
> useful.

**Q: Why `torch.no_grad()` around inference?**
> A: Inference never needs gradients — `no_grad()` disables autograd's
> bookkeeping (which exists to support `.backward()`), saving memory and
> some compute. Training's `train_one_epoch` does need gradients, so it
> doesn't use `no_grad()`; `evaluate()` does, since it's a forward-pass-only
> validation loop, same as inference.

---

## F. Docker

**Q: What does "multi-stage build" mean here, and what's actually gained
by it?**
> A: Both Dockerfiles have a `base` stage (installs pinned Python deps)
> and a role-specific final stage (copies in only that role's source
> files). The gain isn't dropping build tooling here specifically (both
> stages use the same slim Python base) — it's that Docker's layer cache
> means the (slow) `pip install` layer is unaffected by source-code
> changes, since `requirements/*.txt` is copied and installed *before*
> `src/` is copied in. Changing application code triggers a fast rebuild
> of just the last couple of layers, not a full dependency reinstall.

**Q: Why does the serving image run as a non-root user but the training
image doesn't?**
> A: Running as root inside a container is a security anti-pattern — if
> the process is compromised, a root process has a larger blast radius
> (e.g. easier container-escape primitives). The serving image is a
> long-running process with an exposed network port, so it drops to
> `appuser` (uid 1000). The training image is a short-lived batch job
> with no exposed port and no long-running server — different risk
> profile, so it's a deliberate, defensible asymmetry rather than an
> inconsistency.

**Q: Why port 8080 instead of port 80 for the serving container?**
> A: Binding to ports below 1024 traditionally requires root privileges
> on Linux. Since the serving container runs as a non-root user, it must
> use an unprivileged port — 8080 is the conventional choice. The
> Kubernetes `Service` translates the conventional external port 80 to
> the pod's actual 8080 internally, so callers of the Service never need
> to know the pod's real listening port.

**Q: What does the `HEALTHCHECK` instruction in `Dockerfile.serve`
actually do, and why doesn't `Dockerfile.train` have one?**
> A: It tells the Docker daemon to periodically run `curl -f
> http://localhost:8080/health` (every 10s, 3s timeout, 15s initial grace
> period, 3 retries before marking unhealthy) and surface the result as
> `docker ps`'s `(healthy)`/`(unhealthy)` status — independent of
> Kubernetes' own probes, useful e.g. for plain `docker run` debugging.
> The training image has no long-running process to poll — it runs to
> completion and exits — so a `HEALTHCHECK` wouldn't have anything
> meaningful to check.

**Q: Why do both `requirements/*.txt` files point at
`https://download.pytorch.org/whl/cpu`?**
> A: PyPI's default `torch` wheel bundles the full CUDA runtime (multiple
> GB) even on CPU-only machines. Pointing at PyTorch's dedicated CPU-only
> wheel index avoids that bloat — it cut the training image from 5.1GB to
> 1.1GB — with no behavior change on CPU-only hosts; `torch.cuda.is_available()`
> still correctly reports `False`.

**Q: What's in `.dockerignore` and why does it matter?**
> A: `.git/`, `data/`, `checkpoints/`, virtualenvs, `__pycache__/`,
> `tests/`, `k8s/`, `*.md`, etc. Two reasons: keeps large/irrelevant files
> (the ~170MB dataset tarball, multi-hundred-MB checkpoints) from ever
> being sent to the Docker daemon as build context, and keeps files
> neither Dockerfile's `COPY` instructions reference out of the context
> entirely.

---

## G. Kubernetes

**Q: Walk through what happens, in order, when you run the full
Kubernetes workflow from scratch.**
> A: `kind create cluster` spins up the cluster; `kind load docker-image`
> pushes both images directly into the cluster's node (bypassing any
> registry, which is kind-specific). Then `kubectl apply` in order:
> `namespace.yaml` (creates `ml-training`), `configmap.yaml` (the training
> hyperparameters), `training-job.yaml` (creates two PVCs and the Job,
> which starts running immediately once its volumes are bound).
> `kubectl wait --for=condition=complete job/...` blocks until training
> finishes and writes the checkpoint to the shared PVC. Only then are
> `serving-deployment.yaml`, `serving-service.yaml`, and `hpa.yaml`
> applied — the serving pods would fail their readiness checks
> indefinitely if a checkpoint didn't exist yet, since `/health` returns
> 503 with no model loaded.

**Q: Why is the checkpoints PVC mounted read-write in the training Job
but read-only in the serving Deployment?**
> A: The training Job is the sole writer — it produces the checkpoint.
> The serving Deployment is a consumer only; mounting it read-only is a
> real filesystem-level safety boundary, not just a convention — a bug in
> `serve.py` that attempted to write there would fail at the mount level,
> not silently succeed and potentially corrupt the artifact multiple
> serving replicas depend on.

**Q: Explain the difference between the training Job's resource spec and
the serving Deployment's.**
> A: The training Job sets `requests == limits` (`cpu: "2", memory: 4Gi`
> both), which is Kubernetes' "Guaranteed" QoS class — appropriate for a
> one-shot job where you want predictable, complete resource allocation
> with no risk of being squeezed mid-run. The serving Deployment sets
> `requests` (500m/1Gi) below `limits` (1 core/2Gi) — "Burstable" QoS —
> appropriate for a service whose load varies with traffic and should be
> able to burst above baseline temporarily.

**Q: What's the difference between `livenessProbe` and `readinessProbe`
in the serving Deployment, given they hit the same endpoint?**
> A: Liveness answers "is this container broken enough to need a
> restart" — if `/health` fails 3 consecutive checks (10s apart, so ~30s),
> Kubernetes kills and restarts the pod. Readiness answers "should the
> Service route traffic to this pod right now" — a pod failing readiness
> is pulled out of load-balancing rotation without being killed, giving it
> a chance to recover on its own. Readiness also has a 15s
> `initialDelaySeconds` so it doesn't immediately flag a still-starting
> pod as unready in a way that's treated as unusual — though it correctly
> starts *not-ready* until the model finishes loading in `lifespan`.

**Q: What does `maxSurge: 1, maxUnavailable: 0` actually guarantee during
a rolling update?**
> A: That capacity for serving traffic never drops below the full desired
> replica count at any point during a rollout. Kubernetes may temporarily
> run one extra pod (surge) on the new version while waiting for it to
> become ready, but will never terminate an old, working pod until a new
> one has taken its place — `maxUnavailable: 0` forbids ever having fewer
> than the desired count of *available* pods.

**Q: How does the `HorizontalPodAutoscaler` decide when to scale, and
what does it need from the cluster to work at all?**
> A: It watches average CPU utilization across all pods in the
> `model-serving` Deployment, computed as a percentage of each pod's
> *requested* CPU (500m). Above 70% average utilization it scales up
> (max 5 replicas); sustained well below it, it scales back down (never
> below 2, matching the Deployment's own baseline). It requires the
> `metrics-server` cluster add-on to be installed and actually reporting
> pod CPU metrics — without it, the HPA has no data to act on.

**Q: What is the `training-job-gpu.yaml` variant for, and what
specifically differs from the CPU version?**
> A: A bonus manifest showing how the same training workload would be
> adapted for a real GPU-backed cluster. It adds `nvidia.com/gpu: 1` to
> both `resources.requests`/`limits`, a `nodeSelector:
> {accelerator: nvidia-gpu}` restricting scheduling to GPU-labeled nodes,
> and a toleration for the `nvidia.com/gpu=present:NoSchedule` taint GPU
> node pools are conventionally tainted with. No application code changes
> — `train.py`/`model.py` already select `cuda` automatically via
> `torch.cuda.is_available()` if a GPU is present. It wasn't applied
> during `kind`-based validation because `kind` nodes have no GPUs.

**Q: Why does the training Job set `OMP_NUM_THREADS`/`MKL_NUM_THREADS`
env vars, and what breaks without them?**
> A: PyTorch's intra-op thread pool defaults to `os.cpu_count()`, which
> reports the *host* machine's core count, not the pod's Kubernetes CPU
> quota. On this environment (16-core host, 2-core pod quota), leaving
> this unset caused PyTorch to spin up 16 threads fighting over a 2-core
> cgroup allocation, and the Linux CFS scheduler throttled ~99% of CPU
> periods as a result — the training Job appeared hung for 20+ minutes.
> Setting both env vars to `"2"` (matching the CPU request/limit) fixed
> it: throttling dropped to ~33% and the same Job completed in under 3
> minutes. General lesson: setting `resources.limits.cpu` alone doesn't
> tell the *process* about its quota — any CPU-bound native thread pool
> that defaults to host core count needs an explicit override in
> containerized environments.

**Q: Why two separate `PersistentVolumeClaim`s instead of one shared PVC
for both data and checkpoints?**
> A: Different lifecycles and different consumers. `training-data-pvc` is
> written once by training and never touched again. `model-checkpoints-pvc`
> is written by training but then read by the serving Deployment — it's
> the actual hand-off artifact between the two halves of the pipeline.
> Keeping them separate means the serving Deployment's volume list doesn't
> need to reference the (irrelevant to it) dataset volume at all.

---

## H. CI and tooling

**Q: What does the CI pipeline actually check, and in what order?**
> A: Two jobs: `lint-and-test` (ruff lint, then pytest) runs first;
> `docker-build` (builds both Dockerfile.train and Dockerfile.serve, no
> push) only runs if `lint-and-test` passed (`needs:` dependency). It
> triggers on pushes and PRs against both `main` and `develop`.

**Q: Why does `ruff.toml` restrict `select` to `["E", "F", "W"]` instead
of using ruff's defaults?**
> A: Ruff's default/preview rule set is opinionated well beyond basic
> pycodestyle/pyflakes — it flagged things like `import torch.nn as nn`
> (preferring `from torch import nn`) and FastAPI's idiomatic `File(...)`
> default-argument pattern as issues, which aren't real problems in this
> codebase's conventions. Restricting to `E`(rror)/`F`(pyflakes)/`W`(arning)
> keeps CI enforcing genuinely useful checks (unused imports, undefined
> names, line length) without false-positive failures on stylistic
> preferences the project doesn't follow.

**Q: What do the 8 unit tests in `tests/test_model.py` actually verify?**
> A: Parametrized tests that both `get_model("resnet18", ...)` and
> `get_model("simplecnn", ...)` return the right class and produce
> correctly-shaped output for a batch; a test that `num_classes` is
> respected (output dimension matches); a test that an unknown
> architecture name raises `ValueError`; a checkpoint save/load round-trip
> test that verifies a model reloaded from a saved checkpoint produces
> *identical* output (`torch.allclose`) to the original, unmodified model
> — proving the checkpoint format actually preserves weights correctly;
> and a schema test that `training_config.yaml` has all the keys
> `train.py` expects. None of them run actual training — they test the
> building blocks in isolation, fast enough to run in CI on every push.

---

## I. The two "found in the wild" bugs (likely to come up as its own topic)

**Q: Describe the MKL-DNN bug from start to finish: symptom, diagnosis,
fix.**
> A: Symptom: `Conv2d` forward passes crashed the process with `SIGFPE`
> (exit 136), no Python traceback — plain tensor ops were unaffected.
> Diagnosis: isolated by stepping through ResNet-18 layer-by-layer with
> flushed prints until the crash localized to the first conv layer, then
> reproduced with a bare `nn.Conv2d` on a synthetic tensor to rule out
> anything model/dataset-specific. Several documented env-var workarounds
> for this bug class did nothing. Fix: `torch.backends.mkldnn.enabled =
> False` — a known oneDNN CPU-dispatch bug on some AMD EPYC hosts,
> unrelated to the model or data; disabling that backend routes
> convolutions through a correct, slightly slower path.

**Q: Describe the Kubernetes CPU-throttling bug from start to finish.**
> A: Symptom: the training Job appeared to hang for 20+ minutes on `kind`,
> despite the identical image/config working fine under plain `docker
> run`. Diagnosis: `cat /sys/fs/cgroup/cpu.stat` inside the pod showed
> ~99% of CPU periods throttled; `torch.get_num_threads()` inside the pod
> returned 16 (the host's core count) against a 2-core pod limit. Root
> cause: PyTorch's thread pool sizing ignores Kubernetes CPU quotas by
> default, using `os.cpu_count()` (host-visible) instead. Fix:
> `OMP_NUM_THREADS`/`MKL_NUM_THREADS` env vars pinned to `"2"` in the Job
> spec, matching the CPU request/limit — throttling dropped to ~33% and
> the Job completed in under 3 minutes (versus never finishing before the
> fix).

**Q: What do these two bugs have in common, thematically?**
> A: Both are examples of an environment-level mismatch that's invisible
> until code actually runs somewhere other than a "happy path" dev
> machine — one is a hardware/CPU-instruction-dispatch mismatch, the other
> is an orchestration-layer/resource-quota mismatch. Neither would have
> been caught by unit tests, code review, or reading the manifests —
> only by actually executing the pipeline end-to-end on real
> infrastructure, which is the entire point of this assignment's Part F
> validation requirement.

---

## J. Training run specifics (from the completed full run)

**Q: What are the shipped default hyperparameters?**
> A: `resnet18`, 10 epochs max (early stopping patience 3), batch size 64,
> Adam optimizer at learning rate 0.001, full CIFAR-10 (`subset_fraction:
> 1.0` — all 50,000 training images / 10,000 validation images).

**Q: Did training actually run to the full 10 epochs, or did it stop
early?**
> A: It ran the full 10 epochs and simply hit the epoch cap — early
> stopping (patience 3) got close but never actually triggered. Val_loss
> improved on every epoch through epoch 8 (0.3947, the eventual best),
> then regressed for both epoch 9 (0.4281) and epoch 10 (0.3997) — two
> consecutive non-improving epochs, one short of the patience-3 threshold
> that would have cut the run off before a 10th epoch even started.

**Q: Roughly how long did the full run take, and why?**
> A: About **7 hours** on this CPU-only host, run via plain `docker run`
> with the exact command in the README (not `docker-compose`, not
> Kubernetes — a local container is sufficient to prove the training image
> works end-to-end). It's slower than a typical CPU CIFAR-10 run for two
> compounding reasons: the MKL-DNN workaround (§9a/I) forces convolutions
> through a slower fallback backend, and this is genuinely full-dataset
> (50,000 images), full-epoch-count (10) training, not the 5%-subset,
> 1-epoch `training_config.verify.yaml` used for pipeline smoke tests.

**Q: What accuracy did the final model reach, and which epoch's
checkpoint was actually kept?**
> A: **86.79% validation accuracy**, from the **epoch 8** checkpoint
> (val_loss 0.3947 — the lowest of the run). Epochs 9 and 10 pushed
> accuracy slightly higher (85.71% and 86.91% respectively) but their
> val_loss was *higher* than epoch 8's, so under the "save only on
> val_loss improvement" policy (§D/I) they never overwrote
> `classifier_v1.pt`. This is a good concrete example of why the
> checkpoint policy tracks loss rather than accuracy: accuracy is
> single-threshold-sensitive and can wobble upward even as the model's
> confidence calibration (which loss captures) gets slightly worse.

**Q: How was the resulting checkpoint actually validated, beyond just
looking at the metrics table?**
> A: By building the serving image, mounting the fresh checkpoint into a
> running container, and hitting `/predict` with 6 real CIFAR-10 test
> images pulled straight out of the dataset's `test_batch` (one per class:
> cat, ship, airplane, frog, automobile, truck) — images the model never
> saw during training or validation. Result: 5/6 correct, with >99%
> confidence on cat, ship, frog, and truck. This exercises the full real
> path (checkpoint → `load_model()` → FastAPI → inference), not just the
> training loop's own self-reported metrics.

**Q: What was the one misclassification, and is it a red flag?**
> A: The airplane image was predicted as "ship" (still with high
> confidence). This isn't evidence of a broken pipeline — airplane/ship is
> a well-documented CIFAR-10 confusion pair, since both classes are
> frequently photographed as a small, light-colored, elongated object
> against a plain, uniform background (sky or water) at a similar
> horizontal framing. A single miss out of 6 is exactly what you'd expect
> from an 86.79%-accuracy model, not a symptom of a data, training, or
> serving bug.
