import sys
sys.path.append('..')

import ClusteringExperiment
import cPickle
import gzip
import dill
from operator import attrgetter
from argparse import ArgumentParser
import theano

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
    for k in c._nnet.__dict__.keys():
        if type(c._nnet.__getattribute__(k)) == theano.sandbox.cuda.var.CudaNdarraySharedVariable:
            c._nnet.__setattr__(k,c._nnet.__getattribute__(k).get_value())
    c._nnet.activation_function = None
    c._nnet.output_function = None
    fil = gzip.open(outp, 'wb')
    dill.dump(c,fil)
    fil.close()
