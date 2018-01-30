#!/bin/bash

LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HOME}/my_libc_env/lib/x86_64-linux-gnu/:${HOME}/my_libc_env/usr/lib64/" ${HOME}/my_libc_env/lib/x86_64-linux-gnu/ld-2.17.so ~/anaconda2/bin/python draw_real.py

exit 0
