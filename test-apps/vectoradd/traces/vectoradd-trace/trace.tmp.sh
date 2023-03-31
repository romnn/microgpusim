set -e
export CUDA_INJECTION64_PATH="/home/roman/dev/box/accelsim/accel-sim-framework-dev/util/tracer_nvbit/tracer_tool/tracer_tool.so"
export TRACES_FOLDER="./test-apps/vectoradd/traces/vectoradd-trace"
export USER_DEFINED_FOLDERS="1"
export DYNAMIC_KERNEL_LIMIT_END="0"
export LD_PRELOAD="/home/roman/dev/box/accelsim/accel-sim-framework-dev/util/tracer_nvbit/tracer_tool/tracer_tool.so"
./test-apps/vectoradd/vectoradd
/home/roman/dev/box/accelsim/accel-sim-framework-dev/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing ./test-apps/vectoradd/traces/vectoradd-trace/kernelslist
rm -f ./test-apps/vectoradd/traces/vectoradd-trace/*.trace
rm -f ./test-apps/vectoradd/traces/vectoradd-trace/kernelslist