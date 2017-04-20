import sys
sys.path.append('..')

import ClusteringExperiment
import cPickle
import gzip
import dill
from operator import attrgetter
from argparse import ArgumentParser
from theano import tensor as T

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='input file')
    parser.add_argument('-o', '--output', type=str,
                        help='output file')
    opts = parser.parse_args()
    getter = attrgetter('input', 'output')
    inp, outp = getter(opts)
    fil = gzip.open(inp)
    c = cPickle.load(fil)
    c._nnet.activation_function = T.nnet.sigmoid
    c._nnet.output_function = T.nnet.sigmoid
    fil = gzip.open(outp, 'wb')
    dill.dump(c,fil)
    fil.close()
