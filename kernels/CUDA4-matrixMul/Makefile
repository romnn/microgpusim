CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

NVCCFLAGS   := 
CCFLAGS     :=
LDFLAGS     :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

INCLUDES  := -I../Common
LIBRARIES :=

TEJAS_LIBRARIES := "-L$(TEJAS_ROOT)/../so_files_64bit"
TEJAS_LIBRARIES += "-L$(TEJAS_ROOT)/../"
TEJAS_LIBRARIES += "-L$(CUDA_PATH)/lib64"
TEJAS_LIBRARIES += "-L/usr/lib/x86_64-linux-gnu/"

native: matrixMul.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) --std=c++11 $(GENCODE_FLAGS) -o matrixMul.o -c matrixMul.cu
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o matrixMul matrixMul.o $(LIBRARIES)

m2s: matrixMul.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -m32 -Xcicc -O0 -Xptxas -O0 $(GENCODE_FLAGS) -arch sm_35 -o matrixMul.o -c matrixMul.cu
	$(HOST_COMPILER) $(ALL_LDFLAGS) -L$(M2S_ROOT)/lib/.libs -o matrixMul matrixMul.o -m32 $(LIBRARIES) -lm2s-cuda 

accelsim: matrixMul.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) --std=c++11 $(GENCODE_FLAGS) -o matrixMul.o -c matrixMul.cu
	$(NVCC) $(ALL_LDFLAGS) --cudart shared $(GENCODE_FLAGS) -o matrixMul matrixMul.o $(LIBRARIES)

tejas: matrixMul.cu
	python3 ../inject-main.py matrixMul.cu matrixMulTejas.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -arch=sm_20 -o matrixMul.o -c matrixMulTejas.cu
	g++-4.8 -std=c++0x $(TEJAS_ROOT)/../Tracegen.cpp -c -I$(TEJAS_ROOT)/../
	g++-4.4 -o tracegen matrixMul.o Tracegen.o $(TEJAS_LIBRARIES) -locelot -ltinfo -Wl,-rpath-link=$(TEJAS_ROOT)/../so_files_64bit

macsim: matrixMul.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -arch=sm_20 -o matrixMul.o -c matrixMul.cu
	g++-4.4 -o matrixMul matrixMul.o -locelot -locelotTrace
	python3 ../get-kernel-info.py $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -arch=sm_20 matrixMul.cu

clean:
	rm -f matrixMul matrixMul.o matrixMulTejas.cu

clobber: clean


