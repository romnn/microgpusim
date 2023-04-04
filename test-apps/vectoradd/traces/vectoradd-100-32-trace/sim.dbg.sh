set -e
cd /home/roman/dev/box/accelsim/gtx1080
SIM_PATH='/home/roman/dev/box/accelsim/accel-sim-framework-dev/gpu-simulator/'
source /home/roman/dev/box/accelsim/accel-sim-framework-dev/gpu-simulator/setup_environment.sh

# b gpgpu_sim_thread_concurrent
# gdb -x commands.txt --batch --args executablename arg1 arg2 arg3

gdb --args /home/roman/dev/box/accelsim/accel-sim-framework-dev/gpu-simulator/bin/release/accel-sim.out \
    "-trace" "/home/roman/dev/box/test-apps/vectoradd/traces/vectoradd-100-32-trace/kernelslist.g" \
    "-config" "/home/roman/dev/box/accelsim/gtx1080/gpgpusim.config" \
    "-config" "/home/roman/dev/box/accelsim/gtx1080/gpgpusim.trace.config"
