# Implementation Notes

These notes explain the main choices in the repository. They are intentionally
shorter than the README and focus on decisions that are likely to come up in a
code review.

## Model And Data

The default model is ResNet-18 adapted for 32x32 CIFAR-10 images. The stock
ImageNet stem is too aggressive for CIFAR-10, so src/model.py replaces the
7x7 stride-2 convolution with a 3x3 stride-1 convolution and removes the first
max-pool. A smaller SimpleCNN is also available for quick checks.

src/dataset.py uses torchvision's CIFAR-10 loader instead of custom parsing.
Training uses random crop and horizontal flip augmentation; validation and
serving use deterministic tensor conversion and CIFAR-10 normalization. The
same eval transform is imported by src/serve.py, which avoids drift between
validation and inference preprocessing.

The optional subset_fraction config field is only for fast validation runs.
The submitted default config keeps subset_fraction at 1.0 and trains on the
full CIFAR-10 training split.

## Training

src/train.py resolves the config in this order:

1. TRAINING_CONFIG_PATH environment variable.
2. /app/configs/training_config.yaml, used by Docker and Kubernetes.
3. configs/training_config.yaml, used for local development.

The training loop logs JSON lines to stdout, saves the checkpoint with the
lowest validation loss, and stops early when validation loss fails to improve
for the configured patience. The checkpoint stores architecture and class
count along with the model weights, so the serving app can reconstruct the
right model without relying on the current config file.

## Serving

src/serve.py is a FastAPI app with two assignment-required endpoints:

- GET /health returns 200 only after a checkpoint is loaded.
- POST /predict accepts a multipart image file and returns the predicted class
  plus CIFAR-10 class probabilities.

The checkpoint path is read from CHECKPOINT_PATH, falling back to
/app/checkpoints/classifier_v1.pt and then the local checkpoints/ directory.

## Docker

docker/Dockerfile.train installs only training requirements, copies src/ and
configs/, sets TRAINING_CONFIG_PATH, and uses python src/train.py as the
entrypoint. It expects /app/data and /app/checkpoints to be mounted so dataset
downloads and model checkpoints persist outside the container.

docker/Dockerfile.serve installs inference requirements, copies only the serving
code path, exposes port 8080, runs as a non-root user, and includes a
HEALTHCHECK against /health.

Both requirement files pin versions and use the CPU-only PyTorch wheel index.
That keeps the images much smaller than the default CUDA-enabled PyPI wheels
for this CPU-based assignment environment.

## Kubernetes

The manifests use the ml-training namespace. k8s/configmap.yaml provides the
training config and k8s/training-job.yaml mounts it at /app/configs. The Job
also mounts PVCs at /app/data and /app/checkpoints and sets the required CPU
and memory requests/limits.

k8s/serving-deployment.yaml runs two replicas, mounts the checkpoint PVC
read-only, configures /health liveness/readiness probes, sets resource
requests/limits, and uses a rolling update with maxSurge 1 and maxUnavailable
0. k8s/serving-service.yaml exposes the app as a ClusterIP service on port 80.
k8s/hpa.yaml scales the Deployment from 2 to 5 replicas at 70 percent CPU.

k8s/training-job-gpu.yaml is the bonus manifest. It adds nvidia.com/gpu: 1,
a GPU node selector, and an NVIDIA toleration for clusters with GPU nodes.

## Validation Notes

Two environment-specific issues came up during validation:

- MKL-DNN convolution crashed with SIGFPE on the local AMD EPYC CPU. Disabling
  torch.backends.mkldnn in src/model.py fixed model construction, tests,
  training, and serving on this host.
- The Kubernetes training pod was heavily CPU-throttled because PyTorch used
  the host CPU count instead of the pod's 2-core limit. The Job sets
  OMP_NUM_THREADS=2 and MKL_NUM_THREADS=2 to match the CPU quota.

VALIDATION.md contains the terminal evidence to paste into the final PR body.
