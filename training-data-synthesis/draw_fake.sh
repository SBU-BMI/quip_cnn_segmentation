#!/bin/bash
# Start 5 processes

nohup python draw_fake.py 0 5 &
nohup python draw_fake.py 1 5 &
nohup python draw_fake.py 2 5 &
nohup python draw_fake.py 3 5 &
nohup python draw_fake.py 4 5 &

exit 0
