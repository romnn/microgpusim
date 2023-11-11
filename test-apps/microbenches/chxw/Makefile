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
# 86_EXECUTABLES := $(EXECUTABLES:%=%_86)
CUBINS := $(EXECUTABLES:%=%.cubin)

# Default target: build all executables
all: $(EXECUTABLES) $(DEBUG_EXECUTABLES) $(CUBINS)

86: $(EXECUTABLES:%=%_86)
80: $(EXECUTABLES:%=%_80)
61: $(EXECUTABLES:%=%_61)
35: $(EXECUTABLES:%=%_35)
52: $(EXECUTABLES:%=%_52)
75: $(EXECUTABLES:%=%_75)

# Rule to build each executable
%: %.cu common.cu
	$(NVCC) $(NV_FLAGS) -arch sm_61 $^ -o $@

# Rule to build each debug executable
%_debug: %.cu common.cu
	$(NVCC) $(NV_FLAGS) -arch sm_61 $(NV_DEBUG_FLAGS) $^ -o $@

%_86: %.cu common.cu
	$(NVCC) $(NV_FLAGS) -arch sm_86 $(NV_DEBUG_FLAGS) $^ -o $@

%_80: %.cu common.cu
	$(NVCC) $(NV_FLAGS) -arch sm_80 $(NV_DEBUG_FLAGS) $^ -o $@

%_35: %.cu common.cu
	$(NVCC) $(NV_FLAGS) -arch sm_35 $(NV_DEBUG_FLAGS) $^ -o $@

%_52: %.cu common.cu
	$(NVCC) $(NV_FLAGS) -arch sm_52 $(NV_DEBUG_FLAGS) $^ -o $@

%_75: %.cu common.cu
	$(NVCC) $(NV_FLAGS) -arch sm_75 $(NV_DEBUG_FLAGS) $^ -o $@

%_61: %.cu common.cu
	$(NVCC) $(NV_FLAGS) -arch sm_61 $(NV_DEBUG_FLAGS) $^ -o $@

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
	rm -rf $(EXECUTABLES:%=%_86)
	rm -rf $(EXECUTABLES:%=%_80)
	rm -rf $(EXECUTABLES:%=%_61)
	rm -rf $(EXECUTABLES:%=%_35)
	rm -rf $(EXECUTABLES:%=%_52)
	rm -rf $(EXECUTABLES:%=%_75)
	rm -rf $(DEBUG_EXECUTABLES)

.PHONY: all clean
