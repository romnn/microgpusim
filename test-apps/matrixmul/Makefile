all:
	nvcc -ccbin=$(CXX) -O3 -arch=sm_35 matrixmul.cu -o matrixmul

clean:
	rm -f matrixmul
