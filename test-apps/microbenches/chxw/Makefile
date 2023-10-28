CUDA_PATH ?= /usr/local/cuda
NV_FLAGS = -Xptxas="-O3 -dlcm=ca"
NV_DEBUG_FLAGS = -g -lineinfo

# NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)
# NVCC          := nvcc -ccbin $(CXX)
NVCC          := nvcc -ccbin gcc

# List of all .cu files in the source directory
CUFILES := $(wildcard *.cu)

# Extract the names of the executables from .cu filenames
EXECUTABLES := $(CUFILES:%.cu=%)
EXECUTABLES := $(filter-out common, $(EXECUTABLES))
DEBUG_EXECUTABLES := $(EXECUTABLES:%=%_debug)
AMPERE_EXECUTABLES := $(EXECUTABLES:%=%_86)
CUBINS := $(EXECUTABLES:%=%.cubin)

# Default target: build all executables
all: $(EXECUTABLES) $(AMPERE_EXECUTABLES) $(DEBUG_EXECUTABLES) $(CUBINS)

# Rule to build each executable
%: %.cu common.cu
	$(NVCC) $(NV_FLAGS) -arch sm_61 $^ -o $@

# Rule to build each debug executable
%_debug: %.cu common.cu
	$(NVCC) $(NV_FLAGS) -arch sm_61 $(NV_DEBUG_FLAGS) $^ -o $@

# Rule to build each debug executable
%_86: %.cu common.cu
	$(NVCC) $(NV_FLAGS) -arch sm_86 $(NV_DEBUG_FLAGS) $^ -o $@

# Rule to build each cubin
%.cubin: %.cu common.cu
	$(NVCC) $(NV_FLAGS) -arch sm_61 -cubin -dlink $^ -o $@

# Clean build directory
clean:
	rm -rf *.o
	rm -rf *.dsass
	rm -rf *.cubin
	rm -rf *.cuasm
	rm -rf $(EXECUTABLES)
	rm -rf $(AMPERE_EXECUTABLES)
	rm -rf $(DEBUG_EXECUTABLES)

.PHONY: all clean
