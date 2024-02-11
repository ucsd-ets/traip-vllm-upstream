# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

#################### BASE BUILD IMAGE ####################
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS dev

RUN apt-get update -y \
    && apt-get install -y python3-pip git

WORKDIR /workspace

RUN pip install -U xformers==0.0.22.post7+cu118 torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# install build and runtime dependencies
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# install development dependencies
COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt
#################### BASE BUILD IMAGE ####################


#################### EXTENSION BUILD IMAGE ####################
FROM dev AS build

# install build dependencies
COPY requirements-build.txt requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-build.txt

# copy input files
COPY csrc csrc
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY vllm/__init__.py vllm/__init__.py

# cuda arch list used by torch
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
# max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads
# make sure punica kernels are built (for LoRA)
ENV VLLM_INSTALL_PUNICA_KERNELS=1

RUN python3 setup.py build_ext --inplace
#################### EXTENSION Build IMAGE ####################

