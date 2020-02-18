#!/usr/bin/env bash

{
sleep 5
python3 test_model_complex.py \
--GPU_id 0
}&

python3 test_model_complex.py \
--GPU_id 1

wait