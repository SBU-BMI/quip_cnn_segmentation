#!/bin/bash

python extract_all.py
cd new_data_400x400
python data_aug.py

exit 0
