CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER ?= $(CXX)
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

all:
	$(NVCC) -lineinfo -Xptxas="-O3 -dlcm=ca" matrixmul.cu -o matrixmul
	
clean:
	rm -f matrixmul
