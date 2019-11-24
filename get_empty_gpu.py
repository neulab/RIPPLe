import subprocess
import sys
import os
import pandas as pd

def argmax(itr):
    max_ = None
    arg = None
    for i,e in enumerate(itr):
        if i == 0:
            arg,max_ = i,e
        else:
            if e > max_:
                arg,max_ = i,e
    return arg

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return argmax(memory_available)

free_gpu_id = get_freer_gpu()
print(free_gpu_id)
