#!/bin/sh

set -e

DIR=$(dirname "$0")
cd $DIR/../

CONTAINER_NAME=trace
CONTAINER_TAG=latest
CONTAINER="${CONTAINER_NAME}:${CONTAINER_TAG}"
# OUT="$DIR/trace.tar.gz"
TAR_OUT="$DIR/trace/trace.tar"
SCF_OUT="$DIR/trace/trace.scf"

time sudo singularity build --force ./trace/trace.sif ./trace/trace.def

# build container
docker build --progress plain -t "${CONTAINER_NAME}" -f $DIR/Dockerfile .
# docker buildx build --output type=oci -t "${CONTAINER_NAME}" -f $DIR/Dockerfile .

# get container ID
CONTAINER_ID=$(docker inspect --format '{{.ID}}' "${CONTAINER}")
CONTAINER_SIZE=$(docker inspect --format '{{.Size}}' "${CONTAINER}")
echo "built ${CONTAINER} (${CONTAINER_ID}) of size ${CONTAINER_SIZE} bytes"

# build SIF
D2S_IMAGE="quay.io/singularity/docker2singularity:v3.10.5"
docker run \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $DIR/trace:/output \
    --privileged \
    -t --rm ${D2S_IMAGE} \
    --name trace \
    ${CONTAINER}

# save container to tar archive
# echo "saving ${CONTAINER} (${CONTAINER_ID}) to $TAR_OUT"
# docker image save "${CONTAINER}" -o $TAR_OUT

# echo "converting to singularity ${CONTAINER} (${CONTAINER_ID}) to ${SCF_OUT}"
# singularity build ${SCF_OUT} docker-archive://${TAR_OUT}

# DEPRECATED
#
# docker image save "${CONTAINER}" | pigz --fast > "$OUT"
# docker save "${CONTAINER_ID}" -o ./trace.tar
# docker save "${CONTAINER_ID}" | tqdm --bytes --total $CONTAINER_SIZE > $DIR/trace.tar
