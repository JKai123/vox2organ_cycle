#!/bin/bash
source /opt/conda/bin/activate miccai && python3 main.py \
    --train \
    --proj=corticalflow \
    --dataset OASIS_FS72 \
    --exp_prefix lrz-exp_ \
    --group "CorticalFlow no-patch step 1" "CorticalFlow no-patch step 2" "CorticalFlow no-patch step 3"
