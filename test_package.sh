#!/bin/bash

export OPENBLAS_NUM_THREADS=1
set -e
PACKAGE_NAME=replay
. ./venv/bin/activate
cd docs
mkdir -p _static
make clean html
cd ..
pycodestyle --ignore=E203,E231,E501,W503,W605 --max-doc-length=160 ${PACKAGE_NAME} tests
pylint --rcfile=.pylintrc ${PACKAGE_NAME}
#mypy --ignore-missing-imports ${PACKAGE_NAME} tests
pytest --cov=${PACKAGE_NAME} --cov-report=term-missing \
       --doctest-modules ${PACKAGE_NAME} --cov-fail-under=93 tests
