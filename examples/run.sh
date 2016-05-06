#!/bin/sh
source clean.sh
python example_simulate.py
chromium */*png
