# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

####
f = open('bench_labwork1_teamsize.txt')
lines = [line for line in f]
f.close()

x = []
y = []
for line in lines:
    line = [float(x) for x in line.split()]
    x.append(line[0])
    y.append(line[1])
    
plt.plot(x,y)
plt.xlabel('Number of threads')
plt.ylabel('Run time (ms)')
plt.title('Effect of team size')
plt.savefig('report1_teamsize.pdf')
plt.close()

######
f = open('bench_labwork1_dynamic.txt')
lines = [line for line in f]
f.close()

x = []
y = []
for line in lines:
    line = [float(x) for x in line.split()]
    x.append(line[0])
    y.append(line[1])
    
plt.plot(x,y)
plt.xlabel('Number of pixel-block')
plt.ylabel('Run time (ms)')
plt.title('Effect of dynamic schdules with different number of blocks')
plt.savefig('report1_dynamic_schedule.pdf')
plt.close()