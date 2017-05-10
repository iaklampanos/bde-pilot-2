#!/usr/bin/env python

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import sys


def sisint(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
    
def main():
    fname = sys.argv[1]
    ints = []
    count = 0
    for l in open(fname):
        l = l.strip()
        if sisint(l):
            count += 1
            ints.append(int(l))
    ints = np.array(ints)
    n, bins, patches = plt.hist(ints, bins=range(1,21), normed=1, facecolor='black', alpha=0.75)
    plt.axis([1, 21, 0, .15])
    plt.show()
    print 'Read', count, 'lines.'
    
if __name__ == '__main__':
    main()