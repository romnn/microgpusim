# FROM ubuntu:22.04 as base
# FROM rust:1.67 as base
# FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as BASE
FROM nvidia/cuda:11.8.0-devel-ubuntu18.04 as BASE

RUN apt-get update && apt-get install -y \
    build-essential \
    curl

# install rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# force updating the crates registry
RUN cargo install lazy_static; exit 0

# accelsim dependencies
RUN apt-get install -y \
  wget build-essential xutils-dev bison zlib1g-dev flex \
  libglu1-mesa-dev libssl-dev libxml2-dev libboost-all-dev git g++

ENV CUDA_INSTALL_PATH=/usr/local/cuda

# add source
# WORKDIR /app
# COPY . /app

# RUN bash /app/accelsim/build.sh
# python-setuptools python3-dev python3-pip
ENTRYPOINT [ "/bin/bash" ]
# RUN make -C /app/accelsim/accel-sim-framework-dev/gpu-simulator
# compile

# RUN cargo build -p accelsim
