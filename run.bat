@echo off
echo Extracting training data...
powershell -command "Expand-Archive -Path 'training_data.zip' -DestinationPath 'train_data'"

echo Running training script...
python train.py --train_path train_data

echo Done.
pause
