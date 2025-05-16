#!/bin/bash

# using same dataset as https://github.com/hamishivi/automated-instruction-selection/blob/main/shell_scripts/download_train_data.sh
wget https://huggingface.co/datasets/hamishivi/lsds_data/resolve/main/eval.zip
unzip eval.zip
mkdir -p data
mv eval/ data/eval/