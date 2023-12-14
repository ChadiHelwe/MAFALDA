#!/bin/bash

# List of valid models
valid_models=("LLaMA2" "LLaMA2-Chat" "LLaMA2-Instruct" "Falcon" "Mistral" "Mistral-Instruct" "Vicuna" "WizardLM" "Zephyr")

# Default values
default_model_index="1"
default_gpus="0"
default_quantization="4-bit"
default_size="7B"
default_gpu_layers="100"

# Display list of models and read selection
echo "Select a model from the list below by entering its number:"
for i in "${!valid_models[@]}"; do
    echo "$((i+1)). ${valid_models[$i]}"
done

while true; do
    read -p "Enter number (default: $default_model_index): " model_index
    model_index=${model_index:-$default_model_index}
    if [[ "$model_index" =~ ^[0-9]+$ ]] && [ "$model_index" -ge 1 ] && [ "$model_index" -le "${#valid_models[@]}" ]; then
        break
    else
        echo "Invalid number. Please enter a number between 1 and ${#valid_models[@]}"
    fi
done

# Get selected model
modelname=${valid_models[$((model_index-1))]}

# Read other inputs with default values
read -p "Enter GPU devices (default: $default_gpus): " gpus
gpus=${gpus:-$default_gpus}

read -p "Enter quantization (default: $default_quantization): " quantization
quantization=${quantization:-$default_quantization}

read -p "Enter size (default: $default_size): " size
size=${size:-$default_size}

read -p "Enter number of GPU layers (default: $default_gpu_layers): " gpu_layers
gpu_layers=${gpu_layers:-$default_gpu_layers}

# Export GPU devices to be used by CUDA
export CUDA_VISIBLE_DEVICES=$gpus

# Command construction
cmd="python cli.py --model $modelname --size $size --quantization $quantization"

# Iterate over levels
for level in {2..2}
do
    echo "Running $modelname experiments with size $size, quantization $quantization at level $level with $gpu_layers GPU layers on devices $gpus"
    $cmd --level $level --n_gpu_layers $gpu_layers
done
