# -flcm (force load cache modifier)
# -fscm (force store cache modifier)
# -dlcm (default load cache modifier)
# -dscm (defualt store cache modifier)
# cg = cache global (only L2)
# ca = all (L1 + L2)
# cv = volatile (consider cached system memory lines stale fetch again).

# .ca Cache at all levels, likely to be accessed again.
# .cg Cache at global level (cache in L2 and below, not L1).
# .cs Cache streaming, likely to be accessed once.
# .cv Cache as volatile (consider cached system memory lines stale, fetch again).

all: microbench

microbench:
	nvcc gpu.cu hash_function.cpp reverse_engineering.cpp -o microbench
	
# -ccbin=$(CXX) -lineinfo -Xptxas="-O3" 
# nvcc -ccbin=$(CXX) -lineinfo -Xptxas="-O3" gpu.cu hash_function.cpp reverse_engineering.cpp -o microbench

# nvcc -ccbin=$(CXX) -lineinfo -Xptxas="-O3 -flcm=ca" vectoradd.cu -o vectoradd_l1_enabled
# vectoradd_l1_disabled:
# 	nvcc -ccbin=$(CXX) -lineinfo -Xptxas="-O3 -flcm=cg -fscm=cg" vectoradd.cu -o vectoradd_l1_disabled
#
# clean:
# 	rm -f vectoradd_l1_enabled
# 	rm -f vectoradd_l1_disabled
