# mlops-pytorch-pipeline

PyTorch CIFAR-10 training and serving pipeline for the MLOps assignment. The
repo contains the model code, Docker images, Kubernetes manifests, CI checks,
and validation evidence needed for submission.

GitHub repository:
https://github.com/da25g506-dev/mlops-pytorch-pipeline

## Assignment Coverage

| Part | Requirement | Where it is implemented |
| --- | --- | --- |
| A | Repo structure, .gitignore, CI, PR workflow | Project root, .github/workflows/ci.yml, merged PRs #1-#5 |
| B | PyTorch model, dataset, training, serving API | src/model.py, src/dataset.py, src/train.py, src/serve.py |
| C | Training and serving Docker images | docker/Dockerfile.train, docker/Dockerfile.serve, requirements/ |
| D | Kubernetes training Job with ConfigMap and PVCs | k8s/namespace.yaml, k8s/configmap.yaml, k8s/training-job.yaml |
| E | Kubernetes serving Deployment and Service | k8s/serving-deployment.yaml, k8s/serving-service.yaml, k8s/hpa.yaml |
| F | End-to-end validation evidence | VALIDATION.md and final PR body |

## Architecture

~~~mermaid
flowchart LR
    config[training_config.yaml] --> train[src/train.py]
    data[(CIFAR-10 data)] --> train
    model[src/model.py] --> train
    train --> ckpt[(classifier_v1.pt)]
    ckpt --> serve[src/serve.py FastAPI]
    serve --> health[GET /health]
    serve --> predict[POST /predict]

    subgraph Docker
      train_image[mlops-train:v1]
      serve_image[mlops-serve:v1]
    end

    train --> train_image
    serve --> serve_image

    subgraph k8s_cluster[Kubernetes namespace: ml-training]
      cm[ConfigMap]
      job[Training Job]
      pvc_data[(training-data-pvc)]
      pvc_ckpt[(model-checkpoints-pvc)]
      deploy[model-serving Deployment: 2 replicas]
      svc[model-serving Service: 80 to 8080]
      hpa[HPA: 2-5 replicas, 70 percent CPU]
    end

    cm --> job
    pvc_data --> job
    job --> pvc_ckpt
    pvc_ckpt --> deploy
    deploy --> svc
    hpa --> deploy
~~~

## Repository Layout

~~~text
.
|-- src/
|   |-- model.py       # ResNet-18 adapted for 32x32 images, plus SimpleCNN
|   |-- dataset.py     # CIFAR-10 DataLoaders and train/eval transforms
|   |-- train.py       # YAML-driven training loop with JSON-line metrics
|   +-- serve.py       # FastAPI service with /health and /predict
|-- configs/
|   |-- training_config.yaml
|   +-- training_config.verify.yaml
|-- docker/
|   |-- Dockerfile.train
|   +-- Dockerfile.serve
|-- k8s/
|   |-- namespace.yaml
|   |-- configmap.yaml
|   |-- training-job.yaml
|   |-- training-job-gpu.yaml
|   |-- serving-deployment.yaml
|   |-- serving-service.yaml
|   +-- hpa.yaml
|-- requirements/
|   |-- train.txt
|   +-- serve.txt
|-- scripts/
|   +-- make_test_image.py
|-- tests/
|   +-- test_model.py
|-- EXPLANATION.md
+-- VALIDATION.md
~~~

## Local Setup

Python is only needed if you want to run tests or helper scripts outside
Docker.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/train.txt -r requirements/serve.txt
~~~

Run the local checks:

~~~bash
ruff check src/ tests/ scripts/
pytest tests/ -v
~~~

## Docker Workflow

Build the training image:

~~~bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
~~~

Run training with mounted volumes. This uses the assignment config in
configs/training_config.yaml by default and writes the checkpoint to the
host-mounted checkpoints/ directory.

~~~bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-train:v1
~~~

For a quicker infrastructure check, mount the verification config over the
default config:

~~~bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  -v "$(pwd)/configs/training_config.verify.yaml:/app/configs/training_config.yaml:ro" \
  mlops-train:v1
~~~

Build and run the serving image:

~~~bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 \
  -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
  mlops-serve:v1
~~~

Create a small CIFAR-10 image for the prediction request:

~~~bash
python scripts/make_test_image.py --output test_image.png
~~~

Test the API:

~~~bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
~~~

## Kubernetes Workflow

The manifests assume mlops-train:v1 and mlops-serve:v1 are available to the
cluster. For kind, load the locally built images first:

~~~bash
kind create cluster --name mlops-assignment
kind load docker-image mlops-train:v1 --name mlops-assignment
kind load docker-image mlops-serve:v1 --name mlops-assignment
~~~

Apply the training layer:

~~~bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml
kubectl wait --for=condition=complete job/cifar10-training-job -n ml-training --timeout=1200s
~~~

Apply the serving layer:

~~~bash
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
~~~

Verify and test:

~~~bash
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
~~~

k8s/training-job-gpu.yaml is the optional GPU variant. It requests
nvidia.com/gpu: 1, adds a GPU node selector, and includes a toleration for
the usual NVIDIA GPU taint.

## Validation Evidence

Use VALIDATION.md as the source for the final PR evidence. It includes:

- local lint and test output,
- Docker build/train/serve output,
- Kubernetes apply/status/port-forward/curl output,
- full 10-epoch training metrics from the default config,
- the GitHub CI status seen for the public repo.

## Git Workflow

The local history shows the required branch pattern:

- main and develop exist.
- Feature branches were merged through PRs:
  - #1 feature/repo-scaffold
  - #2 feature/pytorch-model
  - #3 feature/docker-training
  - #4 feature/k8s-deployment
- Initial release PR #5 merged the implementation into main.
- Final validation release PR #13 merged the submission hardening into main.
- Later documentation PRs #6-#11 kept the same PR-based workflow.

Use Conventional Commit style for any follow-up commit. Because this cleanup
uses AI assistance, the commit message should cite that, for example:

~~~text
docs: tighten assignment evidence and submission notes (AI-assisted)
~~~
