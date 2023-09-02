CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER ?= $(CXX)
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
#
# NVCCFLAGS   := -O0 -g
# CCFLAGS     :=
# LDFLAGS     :=
#
# ALL_CCFLAGS :=
# ALL_CCFLAGS += $(NVCCFLAGS)
# ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
# ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
# ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))
#
# ALL_LDFLAGS :=
# ALL_LDFLAGS += $(ALL_CCFLAGS)
# ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
# ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

INCLUDES += ../cuda-samples-10.2/Common/

all: transpose

# transpose: transpose.cu
# 	$(NVCC) -I$(INCLUDES) $(ALL_CCFLAGS) --std=c++11 $(GENCODE_FLAGS) -o transpose.o -c transpose.cu
# 	$(NVCC) $(ALL_LDFLAGS) --cudart shared $(GENCODE_FLAGS) -o transpose transpose.o $(LIBRARIES)

transpose: transpose.cu
	$(NVCC) -I$(INCLUDES) -O3 -arch=sm_35 transpose.cu -o transpose

clean:
	rm -f transpose transpose.o

clobber: clean


