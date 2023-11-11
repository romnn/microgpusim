#!/bin/sh

set -e

# try das6 first, then das 5
module load cuda11.7/toolkit || module load cuda11.1/toolkit/11.1.1

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

SCRATCH="/var/scratch/rdm420/"
REPO="$SCRATCH/gpucachesim/"

cd $REPO
git pull
# make -B -C ./test-apps/microbenches/chxw/ pchase_86 set_mapping_86
make $@ -C $REPO/test-apps/microbenches/chxw/
cp $REPO/test-apps/microbenches/chxw/pchase_86 $SCRATCH/pchase
cp $REPO/test-apps/microbenches/chxw/set_mapping_86 $SCRATCH/set_mapping
