CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER ?= g++-8
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

NVCCFLAGS   := 
CCFLAGS     :=
LDFLAGS     :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS := -L/usr/local/cuda/lib64
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

INCLUDES  := 
LIBRARIES := 

TEJAS_LIBRARIES := "-L$(TEJAS_ROOT)/../so_files_64bit"
TEJAS_LIBRARIES += "-L$(TEJAS_ROOT)/../"
TEJAS_LIBRARIES += "-L$(CUDA_PATH)/lib64"
TEJAS_LIBRARIES += "-L/usr/lib/x86_64-linux-gnu/"


native: CUDAStream.cu CUDAStream.h main.cpp Stream.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) --std=c++11 $(GENCODE_FLAGS) -o CUDAStream.o -c CUDAStream.cu
	$(HOST_COMPILER) $(ALL_LDFLAGS) -DCUDA -o CUDAStream main.cpp CUDAStream.o $(LIBRARIES) -lcudart

accelsim: CUDAStream.cu CUDAStream.h main.cpp Stream.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) --std=c++11 $(GENCODE_FLAGS) -o CUDAStream.o -c CUDAStream.cu
	$(NVCC) $(ALL_LDFLAGS) -DCUDA --cudart shared $(GENCODE_FLAGS) -o CUDAStream main.cpp CUDAStream.o $(LIBRARIES)

# Warp.cc:148: unrecognized instruction  pc a8
m2s: CUDAStream.cu CUDAStream.h main.cpp Stream.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -m32 -Xcicc -O0 -Xptxas -O0 $(GENCODE_FLAGS) -arch sm_35 -o CUDAStream.o -c CUDAStream.cu
	$(HOST_COMPILER) $(ALL_LDFLAGS) -L$(M2S_ROOT)/lib/.libs --std=c++11 -DCUDA -o CUDAStream main.cpp CUDAStream.o -m32 $(LIBRARIES) -lm2s-cuda 

tejas: CUDAStream.cu CUDAStream.h main.cpp Stream.h
	python3 ../inject-main.py main.cpp tejasMain.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -arch=sm_20 -o CUDAStream.o -c CUDAStream.cu
	g++-4.8 -std=c++0x $(TEJAS_ROOT)/../Tracegen.cpp -c -I$(TEJAS_ROOT)/../
	g++-4.8 -std=c++11 -DCUDA -o tracegen CUDAStream.o tejasMain.cpp Tracegen.o $(TEJAS_LIBRARIES) -locelot -ltinfo -Wl,-rpath-link=$(TEJAS_ROOT)/../so_files_64bit

macsim: CUDAStream.cu CUDAStream.h main.cpp Stream.h
	python3 ../get-kernel-info.py $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -arch=sm_20 CUDAStream.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -arch=sm_20 -o CUDAStream.o -c CUDAStream.cu
	g++-4.4 -std=c++0x -DCUDA -o CUDAStream main.cpp CUDAStream.o \
		$(LIBRARIES) -Wl,--whole-archive -locelot -locelotTrace -Wl,--no-whole-archive \
		-lz -lGLU -lboost_system -lboost_thread -Wl,-rpath-link=/usr/lib/x86_64-linux-gnu/

clean:
	rm -f CUDAStream *.o

clobber: clean


