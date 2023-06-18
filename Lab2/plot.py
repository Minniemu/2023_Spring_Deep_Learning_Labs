import matplotlib.pyplot as plt
import numpy as np

# parse file 
meanIndexList = []
maxIndexList = []
with open('output3.txt','r') as output:
    lines = output.readlines()
    for s in lines[8:]:
        if 'mean' in s:
            # mean index
            index = s.index('mean')
            # max index
            index_max = s.index('max')
            meanIndexList.append(float(s[index+7:index_max]))
print("mean list = ",meanIndexList)

xpoints = np.array([i for i in range(1000,len(meanIndexList)*1000 + 1,1000)])
ypoints = np.array(meanIndexList)

plt.plot(xpoints, ypoints)
plt.savefig('plot2.png')
