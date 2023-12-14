#!/bin/sh

read -p "Enter OpenAI API key: " api_key

python cli.py --model gpt --api_key $api_key --level 2