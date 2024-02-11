
FROM ghcr.io/ucsd-ets/traip-vllm-upstream:20240211v7 as build

#################### RUNTIME BASE IMAGE ####################
# use CUDA base as CUDA runtime dependencies are already installed via pip
FROM nvidia/cuda:11.8.0-base-ubuntu22.04 AS vllm-base

# libnccl required for ray
RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN pip install -U xformers torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
#################### RUNTIME BASE IMAGE ####################


#################### OPENAI API SERVER ####################
# openai api server alternative
FROM vllm-base AS vllm-openai
# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps accelerate

COPY --from=build /workspace/vllm/*.so /workspace/vllm/
COPY vllm vllm

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
#################### OPENAI API SERVER ####################
