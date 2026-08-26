# Validation Evidence

This file collects the terminal evidence required by Part C and Part F of the
assignment. Use the same content in the final PR description.

## Current Local Checks

Run from the repository root on 2026-08-26.

~~~text
$ ../venv/bin/python -m ruff check src/ tests/ scripts/
All checks passed!
~~~

~~~text
$ ../venv/bin/python -m pytest tests/ -v
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.1.1
collected 10 items

tests/test_model.py::test_get_model_returns_expected_class[resnet18] PASSED
tests/test_model.py::test_get_model_returns_expected_class[simplecnn] PASSED
tests/test_model.py::test_forward_pass_output_shape[resnet18] PASSED
tests/test_model.py::test_forward_pass_output_shape[simplecnn] PASSED
tests/test_model.py::test_get_model_respects_num_classes PASSED
tests/test_model.py::test_get_model_unknown_architecture_raises PASSED
tests/test_model.py::test_checkpoint_save_and_load_roundtrip PASSED
tests/test_model.py::test_training_config_yaml_has_required_keys PASSED
tests/test_serve.py::test_health_and_predict_with_loaded_model PASSED
tests/test_serve.py::test_resolve_checkpoint_path_prefers_existing_env_path PASSED

============================== 10 passed in 6.69s ==============================
~~~

Both Dockerfiles were rebuilt successfully:

~~~text
$ docker build -f docker/Dockerfile.train -t mlops-train:v1 .
#11 writing image sha256:bf024f743fe4194db9652b012855364daf0ff82255b9134254b681310d335ab5 done
#11 naming to docker.io/library/mlops-train:v1 done
#11 DONE 0.0s

$ docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
#12 writing image sha256:6e0243e4341dcbe403a4b0522be33d2d3328ff3b38a6eac3ae4075fb878e05a1 done
#12 naming to docker.io/library/mlops-serve:v1 done
#12 DONE 4.4s

$ docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'
mlops-serve:v1 6e0243e4341d 1.14GB
mlops-train:v1 bf024f743fe4 1.1GB
~~~

All required Kubernetes manifests and the bonus GPU manifest passed a
client-side apply dry-run:

~~~text
$ kubectl apply --dry-run=client --validate=false -f k8s/namespace.yaml
namespace/ml-training unchanged (dry run)

$ kubectl apply --dry-run=client --validate=false -f k8s/configmap.yaml
configmap/training-config configured (dry run)

$ kubectl apply --dry-run=client --validate=false -f k8s/training-job.yaml
persistentvolumeclaim/training-data-pvc unchanged (dry run)
persistentvolumeclaim/model-checkpoints-pvc unchanged (dry run)
job.batch/cifar10-training-job unchanged (dry run)

$ kubectl apply --dry-run=client --validate=false -f k8s/serving-deployment.yaml
deployment.apps/model-serving unchanged (dry run)

$ kubectl apply --dry-run=client --validate=false -f k8s/serving-service.yaml
service/model-serving unchanged (dry run)

$ kubectl apply --dry-run=client --validate=false -f k8s/hpa.yaml
horizontalpodautoscaler.autoscaling/model-serving-hpa unchanged (dry run)

$ kubectl apply --dry-run=client --validate=false -f k8s/training-job-gpu.yaml
persistentvolumeclaim/training-data-pvc unchanged (dry run)
persistentvolumeclaim/model-checkpoints-pvc unchanged (dry run)
job.batch/cifar10-training-job-gpu created (dry run)
~~~

The final main-branch runs also passed:

~~~text
$ gh run list -R da25g506-dev/mlops-pytorch-pipeline --branch main --limit 3
completed success Merge pull request #14 from da25g506-dev/docs/final-pr-link-13 CI main push 32959486524 3m43s
completed success Merge pull request #13 from da25g506-dev/develop CI main push 32956957030 3m53s
completed success Merge pull request #11 from da25g506-dev/develop CI main push 30786541879 4m33s
~~~

## Docker Validation

Recorded in PR #3 and repeated here so the final PR can include the required
terminal log.

~~~text
$ docker build -f docker/Dockerfile.train -t mlops-train:v1 .
... naming to docker.io/library/mlops-train:v1 done
~~~

~~~text
$ docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/configs/training_config.verify.yaml:/app/configs/training_config.yaml:ro" \
    mlops-train:v1

{"event": "config_loaded", "path": "/app/configs/training_config.yaml"}
{"event": "device_selected", "device": "cpu"}
{"event": "epoch_complete", "epoch": 1, "train_loss": 2.0435, "train_accuracy": 0.2492, "val_loss": 4.2691, "val_accuracy": 0.262}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"event": "training_complete", "best_val_loss": 4.2691}
~~~

~~~text
$ ls -la checkpoints/
-rw-r--r-- 1 root root 134217779 classifier_v1.pt
~~~

~~~text
$ docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
... naming to docker.io/library/mlops-serve:v1 done
~~~

~~~text
$ docker run -d --name mlops-serve-test -p 8080:8080 \
    -v "$(pwd)/checkpoints:/app/checkpoints:ro" mlops-serve:v1

$ docker logs mlops-serve-test
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080
~~~

~~~text
$ curl http://localhost:8080/health
{"status":"ok"}   HTTP 200
~~~

~~~text
$ curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
{"predicted_class":"cat","probabilities":{"airplane":0.032548,"automobile":0.027386,"bird":0.123874,"cat":0.410009,"deer":0.051335,"dog":0.137209,"frog":0.128591,"horse":0.042967,"ship":0.027086,"truck":0.018995}}   HTTP 200
~~~

## Kubernetes Validation

Recorded in PR #4 from a local kind cluster.

~~~text
$ kind create cluster --name mlops-assignment
Set kubectl context to "kind-mlops-assignment"

$ kind load docker-image mlops-train:v1 --name mlops-assignment
$ kind load docker-image mlops-serve:v1 --name mlops-assignment
~~~

~~~text
$ kubectl apply -f k8s/namespace.yaml
namespace/ml-training created

$ kubectl apply -f k8s/configmap.yaml
configmap/training-config created

$ kubectl apply -f k8s/training-job.yaml
persistentvolumeclaim/training-data-pvc created
persistentvolumeclaim/model-checkpoints-pvc created
job.batch/cifar10-training-job created
~~~

~~~text
$ kubectl get job -n ml-training
NAME                   STATUS     COMPLETIONS   DURATION   AGE
cifar10-training-job   Complete   1/1           2m56s      4m44s

$ kubectl logs -n ml-training -l role=training
{"event": "config_loaded", "path": "/app/configs/training_config.yaml"}
{"event": "device_selected", "device": "cpu"}
{"event": "epoch_complete", "epoch": 1, "train_loss": 2.0693, "train_accuracy": 0.2288, "val_loss": 3.933, "val_accuracy": 0.224}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"event": "training_complete", "best_val_loss": 3.933}
~~~

~~~text
$ kubectl apply -f k8s/serving-deployment.yaml
deployment.apps/model-serving created

$ kubectl apply -f k8s/serving-service.yaml
service/model-serving created

$ kubectl apply -f k8s/hpa.yaml
horizontalpodautoscaler.autoscaling/model-serving-hpa created
~~~

~~~text
$ kubectl get pods -n ml-training
NAME                          READY   STATUS      RESTARTS   AGE
cifar10-training-job-h6qv8    0/1     Completed   0          6m38s
model-serving-66dd47c-ft4bb   1/1     Running     0          100s
model-serving-66dd47c-kt2ht   1/1     Running     0          100s
~~~

~~~text
$ kubectl describe deployment model-serving -n ml-training
Liveness:     http-get http://:8080/health delay=0s timeout=1s period=10s #success=1 #failure=3
Readiness:    http-get http://:8080/health delay=15s timeout=1s period=5s #success=1 #failure=3
Conditions:
  Available      True    MinimumReplicasAvailable
  Progressing    True    NewReplicaSetAvailable
NewReplicaSet:   model-serving-66dd47c (2/2 replicas created)
~~~

~~~text
$ kubectl get all -n ml-training
NAME                              READY   STATUS      RESTARTS   AGE
pod/cifar10-training-job-h6qv8    0/1     Completed   0          6m38s
pod/model-serving-66dd47c-ft4bb   1/1     Running     0          100s
pod/model-serving-66dd47c-kt2ht   1/1     Running     0          100s

NAME                    TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)   AGE
service/model-serving   ClusterIP   10.96.72.176  <none>        80/TCP    100s

NAME                            READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/model-serving   2/2     2            2           100s

NAME                                                   REFERENCE                  TARGETS          MINPODS   MAXPODS   REPLICAS   AGE
horizontalpodautoscaler.autoscaling/model-serving-hpa  Deployment/model-serving  cpu: <unknown>/70% 2        5         2          100s

NAME                         STATUS     COMPLETIONS   DURATION   AGE
job.batch/cifar10-training-job Complete  1/1           2m56s      6m39s
~~~

The HPA reports unknown current CPU utilization because a default kind cluster
does not include metrics-server. The HPA manifest applied successfully and is
valid for clusters where metrics-server is installed.

~~~text
$ kubectl port-forward svc/model-serving 8080:80 -n ml-training
Forwarding from 127.0.0.1:8080 -> 8080

$ curl http://localhost:8080/health
{"status":"ok"}   HTTP 200

$ curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
{"predicted_class":"airplane","probabilities":{"airplane":0.507454,"automobile":0.158586,"bird":0.040461,"cat":0.006457,"deer":0.091092,"dog":0.007577,"frog":0.016714,"horse":0.001853,"ship":0.152228,"truck":0.017578}}   HTTP 200
~~~

## Full Default Training Run

The default config in configs/training_config.yaml was also run separately with
the full CIFAR-10 training set: ResNet-18, 10 epochs, batch size 64,
subset_fraction 1.0. It took about seven hours on a CPU-only host.

| Epoch | Train loss | Train acc | Val loss | Val acc | Checkpoint saved |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 1.3576 | 50.56% | 1.1535 | 60.97% | yes |
| 2 | 0.9017 | 68.27% | 1.0076 | 65.33% | yes |
| 3 | 0.7100 | 75.18% | 1.1051 | 65.74% | no |
| 4 | 0.5916 | 79.62% | 0.6320 | 79.17% | yes |
| 5 | 0.5168 | 82.14% | 0.5296 | 82.88% | yes |
| 6 | 0.4574 | 84.22% | 0.5266 | 82.18% | yes |
| 7 | 0.4071 | 86.02% | 0.4987 | 84.03% | yes |
| 8 | 0.3722 | 87.15% | 0.3947 | 86.79% | yes, final best-loss checkpoint |
| 9 | 0.3376 | 88.32% | 0.4281 | 85.71% | no |
| 10 | 0.3108 | 89.17% | 0.3997 | 86.91% | no |

The saved checkpoint is from epoch 8 because the training script saves on best
validation loss, not best validation accuracy.
