set -e

DIR=$(dirname "$0")
echo $DIR

source $DIR/accel-sim-framework-dev/gpu-simulator/setup_environment.sh
make -j -C $DIR/accel-sim-framework-dev/gpu-simulator/
