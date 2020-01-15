from sys import stdin
import numpy as np

def getints():
    return map(int , stdin.readline().strip().split())

for t in range(int(input())):
    a, b = getints() #get row and column for H list
    h = list(getints()) #get elements for H list
    np_h = np.asarray(h)
    if np_h.shape != (a, b):
        print("Provide valid array")
        break
    d, e = getints() #get row and column for J list
    j = list(getints())
    np_j = np.asarray(j)
    if np_j.shape != (d, e):
        print("Provide valid array")
        break
    
    for i in np_h:
        for j in i:
            if np.asarray(j).ndim == 2:
                print("Success")
            else:
                print("Fail")
    