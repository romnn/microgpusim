CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

all:
	$(NVCC) -ccbin=$(CXX) -lineinfo -Xptxas="-O3 -dlcm=ca" matrixmul.cu -o matrixmul
	
# $(NVCC) -ccbin=$(CXX) -O3 -arch=sm_35 matrixmul.cu -o matrixmul
# compile nvcc *.cu -o test

clean:
	rm -f matrixmul
