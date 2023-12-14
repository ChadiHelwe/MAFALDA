#!/bin/bash

echo "Running dummy models:
1) Base Silent
2) Base Random
"

read -p "Enter number (default: 1): " selection

if [ "$selection" -eq 1 ]; then
    echo "Running base silent"
    python cli.py --model base-silent --level 2
elif [ "$selection" -eq 2 ]; then
    echo "Running base random"
    python cli.py --model base-random --level 2
else
    echo "Invalid number. Please enter a number between 1 and 2"
fi

