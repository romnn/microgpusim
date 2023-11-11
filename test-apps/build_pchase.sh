#!/bin/sh

module load cuda11.7/toolkit

SCRATCH="/var/scratch/rdm420/"
REPO="$SCRATCH/gpucachesim/"

cd $REPO
git pull
# make -B -C ./test-apps/microbenches/chxw/ pchase_86 set_mapping_86
make $@ -C $REPO/test-apps/microbenches/chxw/
cp $REPO/test-apps/microbenches/chxw/pchase_86 $SCRATCH/pchase
cp $REPO/test-apps/microbenches/chxw/set_mapping_86 $SCRATCH/set_mapping
