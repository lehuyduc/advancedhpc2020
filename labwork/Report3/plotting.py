# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

####
f = open('bench_labwork3_kerneltime.txt')
lines1 = [line for line in f]
f.close()

f = open('bench_labwork3_totaltime.txt')
lines2 = [line for line in f]
f.close()

x = []
kernel = []
total = []
for i in range(len(lines1)):
    line1 = [float(x) for x in lines1[i].split()]
    line2 = [float(x) for x in lines2[i].split()]
    
    x.append(line1[0])
    kernel.append(line1[1])
    total.append(line2[1])
    
plt.plot(x,kernel,label='Kernel only time')
plt.plot(x,total,label='Total time')
plt.legend()
plt.xlabel('Number of threads (block size)')
plt.ylabel('Run time (ms)')
plt.title('Effect of block size')
plt.savefig('report3_kerneltime.pdf')
plt.close()