#!/bin/sh

set -e

SCRATCH="/var/scratch/rdm420/"
REPO="$SCRATCH/gpucachesim/"

if [ ! -d "$SCRATCH" ]; then
    echo "ERROR: $SCRATCH is not a directory."
    echo "Are you running on DAS5 or DAS6?"
    echo ""
    echo "exiting"
    exit 1
fi

# das6
CUDA_11_7=$(module avail cuda11.7/toolkit 2>&1)
# das 5
CUDA_11_1=$(module avail cuda11.1/toolkit 2>&1)

if [ ! -z "$CUDA_11_7" ]; then
    echo "loading CUDA 11.7"
    module load cuda11.7/toolkit
fi
if [ ! -z "$CUDA_11_1" ]; then
    echo "loading CUDA 11.1"
    module load cuda11.1/toolkit
fi

# DAS 5
# -C GTX980
# nodes with an Nvidia GTX980, Maxwell 48 KB L1, 2 MB L2
# -C Titan
# nodes with an Nvidia GTX Titan, Kepler 16 KB L1, 1536 KB L2
# -C K20
# nodes with an Nvidia Tesla K20, Kepler 16 KB L1, 1280 KB L2
# -C RTX2080Ti
# nodes with an Nvidia RTX 2080 Ti, Turing, 64 KB L1, 5.5 MB L2
# -C TitanX
# nodes with an Nvidia GTX TitanX, Maxwell, 48 KB L1, 3 MB L2
# -C TitanX-Pascal
# nodes with an Nvidia GTX TitanX, Pascal, 48 KB L1, 3 MB L2

# we could test TitanX-Pascal, TitanX, GTX980, Titan

cd $REPO
git pull
# make -B -C ./test-apps/microbenches/chxw/ pchase_86 set_mapping_86

for arch in (35 52 61 75 80 86)
do
    echo "building $arch"
    make $@ -C $REPO/test-apps/microbenches/chxw/ $arch
    cp $REPO/test-apps/microbenches/chxw/pchase_$arch $SCRATCH/pchase_$arch
    cp $REPO/test-apps/microbenches/chxw/set_mapping_$arch $SCRATCH/set_mapping_$arch
done
