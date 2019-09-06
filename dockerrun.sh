#!/bin/sh

docker run \
    --rm \
    --volume "$(pwd)":/workspace \
    --gpus all \
    --ipc=host \
    nvcr.io/nvidia/pytorch:19.08-py3 . ./TrainTest.sh