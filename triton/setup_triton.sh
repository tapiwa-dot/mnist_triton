#!/bin/bash

echo "Setting up Triton Model Repository...."

#create directory structure
mkdir -p model_repository/mnist_savedmodel/1

# Train and export the model 
echo "Training MNIST Model..."
python train_and_export.py

# Copy config file 
echo "Copying config file...."
cp config.pbtxt model_repository/mnist_savedmodel/

# Display structure 
echo -e "\n Model Repository Structure....:"
tree model_repository/ || find model_repository/ -print

echo -e "\n Setup Complete!"
echo -e "\n Start Triton Server, run"
echo "docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v \$(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models"