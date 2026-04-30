#!/usr/bin/env python

import glob
import numpy as np
import pandas as pd
import math

def chunker(x, n):
    return [x[i::n] for i in range(n)]

nbatches = 100

filenames = glob.glob("./output-*/summary.tsv")

# read the first file to get some info
df = pd.read_csv(filenames[1], sep="\t",header=None)
recordlength = df[0].str.startswith('#').tolist().index(True) - 1
print("Record length",recordlength)
myrange = list(range(1,int(len(df.index)/recordlength)))
mylist = [x * recordlength+1 for x in myrange]
myrange = list(range(0,recordlength+1))
mylist = myrange + mylist
data = df.drop(mylist).values
rs=len(data)
#print("Data length",rs)
print("Number of files to process",len(filenames))

batched_filenames = chunker(filenames,nbatches)
print("Number of batches",len(batched_filenames))

cnt = 1;
for batch in batched_filenames:

    # read in the data
    print("Reading trajectories in batch",cnt)
    allData = np.empty([0, 12])
    for f in batch:
        df = pd.read_csv(f, sep="\t", header=None)
        df = df.drop(mylist)
        data = df.values  # access the numpy array containing values
        data = data.astype(float)
        allData = np.append(allData, data, axis=0)

    nTraj = int(len(allData)/(recordlength-1))
    print("Found",nTraj,"Trajectories")
    allTrajs = np.split(allData,nTraj)

    meanData = np.mean(allTrajs, axis=0).tolist()
    stderrData = (np.std(allTrajs, axis=0)/math.sqrt(nTraj)).tolist()
    combinedData = [val for pair in zip(meanData, stderrData) for val in pair]

    f = open("summary-avg-"+str(cnt)+".tsv", "w")
    for i in range(len(meanData)):
        for j in range(len(meanData[i])):
            f.write(str(meanData[i][j]))
            if (j>0):
                f.write("\t")
                f.write(str(stderrData[i][j]))
            if (j!=len(meanData[i])-1): f.write("\t")
        f.write("\n")
    f.close()

    cnt = cnt+1
