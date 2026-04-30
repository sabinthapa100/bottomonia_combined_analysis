#!python3

##########################################################
# This python script takes a raw datafile.gz and averages 
# over quantum trajectories, leaving only unique physical 
# trajectories.  It should be executed in the directory
# where the datafile.gz file lives and outputs a new file
# called datafile-avg.gz.
#
#  Usage:  ./processEvents.py
#
##########################################################

import gzip
import math
import numpy as np

d = dict()
dt = np.dtype('double')

cnt = 0
with gzip.open('datafile.gz','r') as f:        
  for line in f:        
    metadata = line.decode('utf-8')
    data = list(map(float,next(f).split()))
    mykey = metadata+str(int(data[-1]))
    d[mykey] = d.get(mykey,[]) + [data]
    cnt += 1
    #if cnt>100:
    #  break
  f.close()

print("Processed", cnt, "trajectory records")
print("Found", len(d), "unique physical trajectories")
print("Performing average over quantum trajectories")

with gzip.open('datafile-avg.gz', 'wt') as f:
  for key in sorted (d.keys()):
    dataArray = np.array(d[key],dtype=dt)
    num = dataArray.shape[0]
    meanData = np.mean(dataArray, axis=0).tolist()[:-2]
    stderrData = (np.std(dataArray, axis=0)/math.sqrt(num)).tolist()[:-2]
    combinedData = [val for pair in zip(meanData, stderrData) for val in pair]
    combinedData = combinedData + [num] + [key[-1]]
    f.write(key[:-1])
    f.write('\t'.join(map(str,combinedData)))
    f.write('\n')
  f.close()
