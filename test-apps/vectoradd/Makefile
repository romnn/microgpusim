# -flcm (force)
# -fscm
# -dlcm (default)
# -dscm
# =cg (disable) =ca (enable)
# cv =	Cache as volatile
#		(consider cached system memory lines stale fetch again).

all:
	nvcc -ccbin=$(CXX) -O3 -arch=sm_35 -lineinfo -Xptxas -dlcm=cg vectoradd.cu -o vectoradd

clean:
	rm -f vectoradd
