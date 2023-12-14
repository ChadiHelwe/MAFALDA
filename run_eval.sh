#!/bin/bash
echo "
1) Models Evaluation
2) Humans Evaluation
"

read -p "Enter number (default: 1): " selection
echo "You selected $selection"

if [ "$selection" -eq 1 ]; then
    echo "Running models evaluation"
    python cli.py --models_eval
elif [ "$selection" -eq 2 ]; then
    echo "Running humans evaluation"
    python cli.py --humans_eval
else
    echo "Invalid number. Please enter a number between 1 and 2"
fi

