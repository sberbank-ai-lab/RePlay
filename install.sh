#!/bin/bash

python3 -m venv venv
source ./venv/bin/activate
python3 resolve_mirror.py
poetry lock
poetry install
