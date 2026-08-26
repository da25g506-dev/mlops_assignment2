# mlops-pytorch-pipeline

CIFAR-10 training and inference with PyTorch, Docker, and Kubernetes.

- Repository: https://github.com/da25g506-dev/mlops-pytorch-pipeline
- Final validation PR: https://github.com/da25g506-dev/mlops-pytorch-pipeline/pull/13
- Terminal evidence: [VALIDATION.md](VALIDATION.md)
- Assignment reflection: [EXPLANATION.md](EXPLANATION.md)

## Architecture

~~~mermaid
flowchart LR
    config[training_config.yaml] --> training[PyTorch training]
    data[(CIFAR-10)] --> training
    training --> checkpoint[(classifier_v1.pt)]
    checkpoint --> api[FastAPI service]
    api --> health[GET /health]
    api --> predict[POST /predict]

    subgraph cluster[Kubernetes: ml-training]
      cm[ConfigMap] --> job[Training Job]
      data_pvc[(Data PVC)] --> job
      job --> model_pvc[(Checkpoint PVC)]
      model_pvc --> deployment[Serving Deployment]
      deployment --> service[ClusterIP Service]
      hpa[HPA] --> deployment
    end
~~~

The default model is ResNet-18 with a smaller input stem for CIFAR-10. Training
reads its settings from YAML, writes JSON metrics to stdout, and saves the
checkpoint with the lowest validation loss. The FastAPI app loads that
checkpoint and exposes health and prediction endpoints.

## Setup

Python 3.10 or newer is required for local testing.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/train.txt -r requirements/serve.txt
pip install pytest==9.1.1 ruff==0.16.4
~~~

Run the checks:

~~~bash
ruff check src/ tests/ scripts/
pytest tests/ -v
~~~

## Docker

Build both images:

~~~bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
~~~

For a quick training check, use the one-epoch verification config:

~~~bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/checkpoints:/app/checkpoints" \
  -v "$PWD/configs/training_config.verify.yaml:/app/configs/training_config.yaml:ro" \
  mlops-train:v1
~~~

Remove the config override to run the full settings from
configs/training_config.yaml.

Start the API:

~~~bash
docker run -d --name mlops-serve-test \
  -p 8080:8080 \
  -v "$PWD/checkpoints:/app/checkpoints:ro" \
  mlops-serve:v1
~~~

Create a test image and call both endpoints:

~~~bash
python scripts/make_test_image.py --output test_image.png
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
~~~

Stop the container when finished:

~~~bash
docker rm -f mlops-serve-test
~~~

## Kubernetes

Create a local cluster and load the images:

~~~bash
kind create cluster --name mlops-assignment
kind load docker-image mlops-train:v1 --name mlops-assignment
kind load docker-image mlops-serve:v1 --name mlops-assignment
~~~

Run training:

~~~bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml
kubectl wait --for=condition=complete \
  job/cifar10-training-job \
  -n ml-training \
  --timeout=1200s
~~~

Deploy the API:

~~~bash
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl get pods -n ml-training
~~~

Test through port forwarding:

~~~bash
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
~~~

The optional k8s/training-job-gpu.yaml requests one NVIDIA GPU. It requires a
cluster with the NVIDIA device plugin and matching node labels.

## Repository Map

- src/: model, data loading, training, and FastAPI code
- configs/: full and quick-check training settings
- docker/: training and serving images
- k8s/: Job, Deployment, Service, ConfigMap, PVC, HPA, and GPU variant
- tests/: model, checkpoint, config, and serving tests
- VALIDATION.md: command output used in the final PR

Development used main, develop, and feature branches. Changes were merged
through pull requests with Conventional Commit messages.
