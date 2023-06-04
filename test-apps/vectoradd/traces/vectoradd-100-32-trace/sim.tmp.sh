set -e
cd /home/roman/dev/box/accelsim/gtx1080
source /home/roman/dev/box/accelsim/accel-sim-framework-dev/gpu-simulator/setup_environment.sh debug
/home/roman/dev/box/target/debug/playground -trace /home/roman/dev/box/test-apps/vectoradd/traces/vectoradd-100-32-trace/kernelslist.g -config /home/roman/dev/box/accelsim/gtx1080/gpgpusim.config -config /home/roman/dev/box/accelsim/gtx1080/gpgpusim.trace.config