# Explanation

GitHub repository:
https://github.com/da25g506-dev/mlops-pytorch-pipeline

Final validation PR:
https://github.com/da25g506-dev/mlops-pytorch-pipeline/pull/13

Validation evidence is collected in VALIDATION.md so the final PR body can
include the same terminal output requested by the assignment.

## Reflection

The most challenging part of this assignment was getting the same training
and serving path to work consistently across three environments: local Python,
Docker, and Kubernetes. The application code is deliberately simple: a
CIFAR-10 classifier, a YAML-driven training loop, and a FastAPI prediction
service. Most of the time went into making the operational path reliable.

The first issue was a CPU-specific PyTorch crash. On this host, convolution
forward passes failed with SIGFPE when the MKL-DNN backend was enabled. The
failure did not produce a normal Python traceback, so I isolated it by testing
the model layer by layer and then reproducing the crash with a bare Conv2d
operation. The fix is in src/model.py: disable MKL-DNN at import time. That
keeps the code slower on this CPU, but it is predictable and works in every
entry point that imports the model.

The second issue appeared only inside the Kubernetes Job. PyTorch sized its
thread pool from the host CPU count, while the pod was limited to two CPU
cores. That caused heavy CFS throttling and made the Job look stuck even
though it was still running. The training manifest now sets OMP_NUM_THREADS
and MKL_NUM_THREADS to 2, matching the Job's CPU request and limit. After that
change the verification Job completed normally on kind.

Docker image size was another practical problem. Installing the default PyPI
torch wheel pulled in CUDA runtime packages that are unnecessary for a
CPU-only training and serving demo. Both requirement files now use the
CPU-only PyTorch wheel index, which keeps the images much smaller while still
satisfying the assignment's PyTorch requirement.

The final result is a pipeline that can be checked in small pieces: unit tests
cover model construction and checkpoint loading, Docker proves the runtime
packaging, and Kubernetes validates the cluster workflow from training Job to
serving Deployment and prediction request.
