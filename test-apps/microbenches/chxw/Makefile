CUDA_PATH ?= /usr/local/cuda
NVFLAGS = -g -lineinfo -Xptxas="-O3 -dlcm=ca"

# NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)
# NVCC          := nvcc -ccbin $(CXX)
NVCC          := nvcc -ccbin gcc

# List of all .cu files in the source directory
CUFILES := $(wildcard *.cu)

# Extract the names of the executables from .cu filenames
EXECUTABLES := $(CUFILES:%.cu=%)

# Default target: build all executables
all: $(EXECUTABLES)

# Rule to build each executable
%: %.cu
	$(NVCC) $(NVFLAGS) $< -o $@

# Clean build directory
clean:
	rm -rf *.o
	rm -rf $(EXECUTABLES)

.PHONY: all clean
