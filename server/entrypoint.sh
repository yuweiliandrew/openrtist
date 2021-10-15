#!/bin/bash
args=$*
source /opt/intel/openvino/bin/setupvars.sh
/usr/bin/nvidia-smi -a
./main.py $args &
./gan_engine_runner.py $args 