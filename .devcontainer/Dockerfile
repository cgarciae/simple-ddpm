FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 AS base
RUN rm -f /etc/apt/sources.list.d/cuda.list && rm -f /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update
RUN apt-get install -y python3 python3-pip python-is-python3

FROM base AS final
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN apt-get install -y python3-tk git
RUN pip install poetry && poetry config virtualenvs.in-project true

FROM base as test
RUN pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python -c "import jax; print(jax.devices())"
