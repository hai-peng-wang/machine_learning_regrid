#!/usr/bin/env bash

set -e

if [ ! -d "${PREFIX}" ]; then
    # yum install wget bzip2 -y
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p ${PREFIX}
    rm miniconda.sh
fi
