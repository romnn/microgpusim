FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
  less \
  vim \
  build-essential \
  pkg-config \
  python3 \
  # fluidGL
  libxi-dev \
  libxmu-dev \
  libglu1-mesa \
  libglu1-mesa-dev \
  freeglut3 \
  freeglut3-dev \
  # for simpleVulkanExample
  libvulkan1 \
  libglfw3 \
  libglfw3-dev \
  # for histEqualizationNPP
  libfreeimage3 \
  libfreeimage-dev \
  # MPI compiler
  mpich

COPY ./benchmarks /app/benchmarks
COPY ./rodinia_3.1 /app/rodinia_3.1
COPY ./cuda-samples-12.4 /app/cuda-samples-12.4
COPY ./generate_ptx.py /app/generate_ptx.py

ENTRYPOINT [ "python3", "generate_ptx.py", "/out" ]

# COPY ./pb2.5benchmarks.tgz /app/pb2.5benchmarks.tgz
# COPY ./rodinia_3.1.tar.bz2 /app/rodinia_3.1.tar.bz2
#
# RUN ls -lia
# RUN tar xvf pb2.5benchmarks.tgz
# RUN tar xvf rodinia_3.1.tar.bz2

# RUN rodinia_3.1
# RUN pbzip2 -d rodinia_3.1.tar.bz2
