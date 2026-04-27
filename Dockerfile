# DROID-SLAM Podman image
# Built and run by SAL's runtime-stress framework via VGGTSLAMAlgorithm-style
# wrapper at src/algorithms/droidslam.py. See docs/DROID_SLAM_PODMAN.md for
# the operator-side build / run / troubleshooting guide.
#
# Notes for future maintainers:
# - DROID-SLAM compiles native CUDA extensions (droid_backends, lietorch,
#   pytorch_scatter); we need a -devel base image with nvcc, not just a
#   runtime image.
# - CUDA 11.8 is the version DROID-SLAM tests against (per install_all.sh).
#   The host NVIDIA driver provides libcuda.so.1 via CDI's nvidia.com/gpu=all
#   bind-mount, and CUDA 11.8 binaries are forward-compat with newer drivers.
# - droid.pth (~16 MB) ships in the repo and is COPYed into the image.
# - Build time: 15-25 minutes on a typical host; one-time per host.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        git build-essential cmake ninja-build pkg-config \
        libeigen3-dev libsuitesparse-dev libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# PyTorch 2.x + torchvision pinned to CUDA 11.8.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
        torch torchvision torchaudio

WORKDIR /droid-slam
COPY . /droid-slam

# Strip prebuilt CUDA artifacts from the host build (they are tied to the
# host's CUDA / glibc and would conflict with the in-container compile).
RUN rm -rf build *.so thirdparty/lietorch/build thirdparty/pytorch_scatter/build

RUN pip install --no-cache-dir -r requirements.txt

# Compile droid_backends + lietorch against CUDA 11.8. torch_scatter has
# a long history of ABI breakage when its bundled source is built against
# newer PyTorch versions (the bundled thirdparty/pytorch_scatter segfaults
# at import under PyTorch 2.7.1+cu118); use the PyG-hosted prebuilt wheel
# matching our exact torch + CUDA combo instead.
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir thirdparty/lietorch \
    && pip install --no-cache-dir torch-scatter \
        -f https://data.pyg.org/whl/torch-2.7.1+cu118.html

# droid_backends.so was linked against PyTorch's libc10/libtorch. The Python
# extension loader doesn't auto-add torch's lib dir to LD_LIBRARY_PATH at
# import time, so set it here so droid_backends imports cleanly at runtime.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}

CMD ["python", "demo.py", "--help"]
