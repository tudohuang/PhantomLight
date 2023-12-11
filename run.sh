#!/bin/bash
echo "Extracting training data..."
unzip -o training_data.zip -d train_data

echo "Running training script..."
python train.py --train_path train_data

echo "Done."
