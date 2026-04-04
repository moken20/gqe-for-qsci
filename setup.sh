#!/bin/bash

device=$1

python3.11 -m venv .env
source .env/bin/activate

pip install --upgrade pip setuptools
if [ "$device" == "gpu" ]; then
    pip install --no-cache-dir -e ".[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128
else
    pip install --no-cache-dir -e ".[cpu]"
fi

pip install ipykernel
python -m ipykernel install --user --name .env --display-name "Python (.env)"