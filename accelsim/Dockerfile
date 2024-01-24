FROM nvidia/cuda:11.8.0-devel-ubuntu18.04 as BASE

ENV CUDA_INSTALL_PATH=/usr/local/cuda

RUN apt-get update && apt-get install -y \
    build-essential \
    curl

# install rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# force updating the crates registry
RUN cargo install lazy_static; exit 0

# # install dependencies
# RUN apt-get install -y llvm-dev libclang-dev clang libssl-dev pkg-config lld
# accelsim dependencies
RUN apt-get install -y \
  wget build-essential xutils-dev bison zlib1g-dev flex \
  libglu1-mesa-dev libssl-dev libxml2-dev libboost-all-dev git g++

WORKDIR /app
COPY ./ /app
RUN ls -liah /app

# RUN make -C /app/accelsim/accel-sim-framework-dev/gpu-simulator

# build the tracer
# tracer: ./accelsim/accel-sim-framework-dev/util/tracer_nvbit/tracer_tool/tracer_tool.so
RUN cargo build -p accelsim-sim
RUN cargo build -p accelsim
RUN cargo build -p accelsim-trace
