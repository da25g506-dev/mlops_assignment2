# Assignment Reflection

- Repository: https://github.com/da25g506-dev/mlops-pytorch-pipeline
- Final validation PR: https://github.com/da25g506-dev/mlops-pytorch-pipeline/pull/13
- Command output: [VALIDATION.md](VALIDATION.md)

I used CIFAR-10 for the project and trained a ResNet-18 model with a modified
input layer for 32x32 images. The training script reads the YAML config, prints
loss and accuracy as JSON, stops when validation loss stops improving, and
saves the best checkpoint. The serving side is a small FastAPI application
with /health and /predict.

The hardest part was not writing the model. It was getting the same code to
behave properly on my machine, in Docker, and inside Kubernetes.

The first problem was a crash during convolution on the local AMD CPU. Python
did not show a traceback; the process exited with SIGFPE. I tested smaller
parts of the model until a plain Conv2d call reproduced it. Disabling MKL-DNN
fixed the crash on this machine, so that setting is applied in src/model.py.

The next problem was the Kubernetes training Job taking far longer than the
same container run directly with Docker. The pod had a two-core CPU limit, but
PyTorch was creating threads based on the host CPU count. The extra threads
spent most of their time being throttled. Setting OMP_NUM_THREADS=2 and
MKL_NUM_THREADS=2 in the Job made the runtime match the pod limit and the test
Job completed normally.

I also found that the default PyTorch wheel included CUDA packages that were
not useful on this CPU-only setup. Using the CPU wheel index reduced the image
sizes without changing the training or inference code.

For validation, I kept two configs. The default config uses all of CIFAR-10
for ten epochs. The verification config uses one epoch and a small subset, so
I could test Docker and Kubernetes without waiting several hours each time.
The default run was still completed separately, and its epoch results are in
VALIDATION.md.

The main lesson from this assignment was that the model is only one part of
the system. Config paths, mounted storage, CPU limits, container users, health
checks, and repeatable test commands all affected whether the application
actually worked after leaving the local Python environment.
