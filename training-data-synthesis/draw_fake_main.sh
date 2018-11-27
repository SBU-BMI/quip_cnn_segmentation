#!/bin/bash

nohup bash draw_fake.sh 0 5 &
sleep 1
nohup bash draw_fake.sh 1 5 &
sleep 1
nohup bash draw_fake.sh 2 5 &
sleep 1
nohup bash draw_fake.sh 3 5 &
sleep 1
nohup bash draw_fake.sh 4 5 &
sleep 1

exit 0
