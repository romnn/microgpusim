CUDA_PATH ?= /usr/local/cuda
NVFLAGS = -g -lineinfo -arch sm_61 -Xptxas="-O3 -dlcm=ca"

# NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)
# NVCC          := nvcc -ccbin $(CXX)
NVCC          := nvcc -ccbin gcc

# List of all .cu files in the source directory
CUFILES := $(wildcard *.cu)

# Extract the names of the executables from .cu filenames
EXECUTABLES := $(CUFILES:%.cu=%)
EXECUTABLES := $(filter-out common, $(EXECUTABLES))
CUBINS := $(EXECUTABLES:%=%.cubin)

# Default target: build all executables
all: $(EXECUTABLES) $(CUBINS)

# Rule to build each executable
%: %.cu common.cu
	$(NVCC) $(NVFLAGS) $^ -o $@

# Rule to build each cubin
%.cubin: %.cu common.cu
	$(NVCC) $(NVFLAGS) -cubin -dlink $^ -o $@

# Clean build directory
clean:
	rm -rf *.o
	rm -rf $(EXECUTABLES)

.PHONY: all clean
