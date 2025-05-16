#!/bin/bash

# Using same dataset as https://github.com/hamishivi/automated-instruction-selection/blob/main/shell_scripts/download_train_data.sh


wget https://huggingface.co/datasets/hamishivi/lsds_data/resolve/main/training_data.zip
unzip training_data.zip
mkdir -p data/training_data
mv training_data/ data/training_data/