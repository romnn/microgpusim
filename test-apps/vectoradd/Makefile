# -flcm (force)
# -fscm
# -dlcm (default)
# -dscm
# =cg (disable) =ca (enable)
# cv =	Cache as volatile
#		(consider cached system memory lines stale fetch again).

# .ca Cache at all levels, likely to be accessed again.
# .cg Cache at global level (cache in L2 and below, not L1).
# .cs Cache streaming, likely to be accessed once.
# .cv Cache as volatile (consider cached system memory lines stale, fetch again).

all: vectoradd_l1_enabled vectoradd_l1_disabled

vectoradd_l1_enabled:
	nvcc -ccbin=$(CXX) -O3 -lineinfo -Xptxas="-dlcm=ca" vectoradd.cu -o vectoradd_l1_enabled

vectoradd_l1_disabled:
	nvcc -ccbin=$(CXX) -O3 -lineinfo -Xptxas="-dlcm=cg" vectoradd.cu -o vectoradd_l1_disabled

clean:
	rm -f vectoradd_l1_enabled
	rm -f vectoradd_l1_disabled
