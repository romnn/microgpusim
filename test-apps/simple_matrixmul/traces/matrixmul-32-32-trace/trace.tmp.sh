set -e
export TRACES_FOLDER="./test-apps/simple_matrixmul/traces/matrixmul-32-32-trace"
export CUDA_INJECTION64_PATH="/home/roman/dev/box/accelsim/accel-sim-framework-dev/util/tracer_nvbit/tracer_tool/tracer_tool.so"
export NOBANNER="1"
export DYNAMIC_KERNEL_LIMIT_END="0"
export LD_PRELOAD="/home/roman/dev/box/accelsim/accel-sim-framework-dev/util/tracer_nvbit/tracer_tool/tracer_tool.so"
export USER_DEFINED_FOLDERS="1"
./test-apps/simple_matrixmul/matrixmul 32 32
/home/roman/dev/box/accelsim/accel-sim-framework-dev/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing ./test-apps/simple_matrixmul/traces/matrixmul-32-32-trace/kernelslist
rm -f ./test-apps/simple_matrixmul/traces/matrixmul-32-32-trace/*.trace
rm -f ./test-apps/simple_matrixmul/traces/matrixmul-32-32-trace/kernelslist