#!/usr/bin/env python

# Given a list of "things" on the stdin, and corresponding parameters, split
# the initial list into train, cross-validation and evaluation sets.
# Usage: cat <file containing list - one item per line> | \
#    div_to_train_test.py <#TRAIN SAMPLES> <#XVAL SAMPLES> <RAND|SEQ> [PREFIX]
#
# It generates 2 or 3 files: PREFIX.train, PREFIX.eval and opt. PREFIX.xval
# Defaults: #XVAL_SAMPLES = 0, SEQ, "out"
#

import sys
import random

def get_rand(things, notrain, noxval):
    things = set(things)
    train = set(random.sample(things, notrain))
    remainder = things - train
    xval = set(random.sample(remainder, noxval))
    remainder = remainder - xval
    return (sorted(train), sorted(xval), sorted(remainder))
    
def get_seq(things, notrain, noxval):
    train = things[:notrain]
    xval = things[notrain:notrain+noxval]
    remainder = things[notrain+noxval:]
    return (train, xval, remainder)

def to_files(divs, prefix):
    for idx, sfx in enumerate(['train', 'xval', 'eval']):
        f = open(prefix + '.' + sfx, 'w')
        for l in divs[idx]:
            f.write(l)
        f.close()

def main():
    noxval = 0
    seltype = 'SEQ'
    prefix = 'samples'
    
    # get the list from the stdin
    things = []
    for l in sys.stdin:
        things.append(l)

    noread = len(things)
    print 'Read ' + str(noread) + ' items from stdin.'

    if len(sys.argv) < 4:
        print 'Usage: cat <file containing list - one item per line> | div_to_train_test.py <#TRAIN SAMPLES> <#XVAL SAMPLES> <RAND|SEQ> [PREFIX]'
        sys.exit(1)
        
    notrain = int(sys.argv[1]) # may raise exception

    noxval = int(sys.argv[2])
    if notrain + noxval >= noread:
        print 'Error: #train + #xval >= #samples...'
        sys.exit(1)

    seltype = sys.argv[3]
    if seltype != 'RAND' and seltype != 'SEQ':
        print 'Error: invalid selection - should be "SEQ" | "RAND"'
        sys.exit(1)

    if len(sys.argv) > 4:
        prefix = sys.argv[4]

    results = None
    if (seltype == 'RAND'):
        results = get_rand(things, notrain, noxval)
    elif (seltype == 'SEQ'):
        results = get_seq(things, notrain, noxval)
    else:
        pass  # shouldn't reach here

    print 'Sizes:', len(results[0]), len(results[1]), len(results[2])

    to_files(results, prefix)
    
    
if __name__ == "__main__":
    main()
