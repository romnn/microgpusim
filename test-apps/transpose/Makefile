CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER ?= $(CXX)
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

INCLUDES += ../cuda-samples-10.2/Common/

all: transpose

transpose: transpose.cu
	$(NVCC) -I$(INCLUDES) -lineinfo -Xptxas="-O3 -dlcm=ca" transpose.cu -o transpose

clean:
	rm -f transpose transpose.o
