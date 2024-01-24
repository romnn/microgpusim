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

# install dependencies
RUN apt-get install -y llvm-dev libclang-dev clang libssl-dev pkg-config lld ncdu

WORKDIR /opt
COPY ./ /opt
RUN ls -liah /opt

# build the tracer
RUN cargo build -p trace
RUN cargo build -p invoke-trace

# ENTRYPOINT [ "/bin/bash" ]
# CMD [ "/app/target/debug/invoke-trace" ]
ENTRYPOINT [ "/opt/target/debug/invoke-trace" ]
