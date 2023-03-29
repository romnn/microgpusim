FROM ubuntu:22.04 as base

RUN apt-get update && apt-get install -y \
    build-essential git curl

RUN apt-get install -y \
  autoconf automake autopoint flex gperf \
  graphviz help2man texinfo valgrind wget rsync

WORKDIR /build
# RUN git clone https://github.com/akimd/bison.git /build/bison
RUN git clone https://git.savannah.gnu.org/git/bison.git /build/bison
WORKDIR /build/bison

# checkout correct tag
# ARG TAG=v2.4.1
ARG TAG=v3.5.1
RUN git checkout tags/$TAG

RUN git submodule update --init
RUN ./bootstrap 

RUN ./configure
RUN make -j
RUN make check
