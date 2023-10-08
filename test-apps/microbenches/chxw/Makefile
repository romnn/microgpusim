CUDA_PATH ?= /usr/local/cuda
NVFLAGS = -g -lineinfo -Xptxas="-O3 -dlcm=ca"

HOST_COMPILER ?= $(CXX)
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

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
