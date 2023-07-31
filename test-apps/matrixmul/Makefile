CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

all:
	$(NVCC) -ccbin=$(CXX) -O3 -arch=sm_35 matrixmul.cu -o matrixmul

clean:
	rm -f matrixmul
