#!/bin/bash

pip3 install virtualenv
python3 -m virtualenv venv
source ./venv/bin/activate
# Прописываем откуда брать пакеты и устанавливаем то, что через поэтри не  ставится
python3 resolve_mirror.py
# TODO: добавить сборку whl и выкладывания в Nexus
poetry lock
poetry install
