CUDA_PATH ?= /usr/local/cuda

NVCC          := $(CUDA_PATH)/bin/nvcc
# -ccbin $(HOST_COMPILER)
# HOST_COMPILER ?= g++-8
#
# NVCCFLAGS   := 
# CCFLAGS     :=
# LDFLAGS     :=
#
# ALL_CCFLAGS :=
# ALL_CCFLAGS += $(NVCCFLAGS)
# ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
# ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
# ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))
#
# ALL_LDFLAGS := -L/usr/local/cuda/lib64
# ALL_LDFLAGS += $(ALL_CCFLAGS)
# ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
# ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))
#
# INCLUDES  := 
# LIBRARIES := 

# CUDAStream: CUDAStream.cu CUDAStream.h main.cpp Stream.h
	# $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) --std=c++11 $(GENCODE_FLAGS) -o CUDAStream.o -c CUDAStream.cu
	# $(HOST_COMPILER) $(ALL_LDFLAGS) -DCUDA -o CUDAStream main.cpp CUDAStream.o $(LIBRARIES) -lcudart

CUDAStream: CUDAStream.cu CUDAStream.h main.cpp Stream.h
	$(NVCC) $(INCLUDES) -O3 -DCUDA -o CUDAStream main.cpp CUDAStream.cu

# accelsim: CUDAStream.cu CUDAStream.h main.cpp Stream.h
# 	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) --std=c++11 $(GENCODE_FLAGS) -o CUDAStream.o -c CUDAStream.cu
# 	$(NVCC) $(ALL_LDFLAGS) -DCUDA --cudart shared $(GENCODE_FLAGS) -o CUDAStream main.cpp CUDAStream.o $(LIBRARIES)

clean:
	rm -f CUDAStream *.o

clobber: clean
