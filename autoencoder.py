import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES'
import sys
import ConfigParser
import numpy as np
import theano
from theano import tensor as T
import datetime
from datetime import datetime
import dataset_utils as utils
import lasagne


def load_config(input_path):
    cp = ConfigParser.ConfigParser()
    cp.read(input_path)
    return cp

def log(s, label='INFO'):
    sys.stderr.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')

def load_data(cp):
    log('Loading data........')
    if cp.get('Experiment','inputfile') == '':
        [X1,labels1] = utils.load_mnist(dataset='training',path='/home/ubuntu/data/mnist/')
        [X2,labels2] = utils.load_mnist(dataset='testing',path='/home/ubuntu/data/mnist/')
        X = np.concatenate((X1,X2),axis=0)
        labels = np.concatenate((labels1,labels2),axis=0)
        X = X.astype(np.float32)*0.02
        np.random.shuffle(X)
        return [X,labels]
    else:
        X = np.load(cp.get('Experiment','inputfile'))
        np.random.shuffle(X)
        return X
    log('DONE........')

def init(cp, dataset):
    # Initialize theano tensors
    log(dataset.shape)
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                              dtype=theano.config.floatX),
                              borrow=True)
    index = T.lscalar()
    # Initialize neural network
    enc_act = cp.get('NeuralNetwork','encoderactivation')
    dec_act = cp.get('NeuralNetwork','decoderactivation')
    relu = lasagne.nonlinearities.rectify
    linear = lasagne.nonlinearities.linear
    batch_size = int(cp.get('NeuralNetwork','batchsize'))
    lr_decay = int(cp.get('NeuralNetwork','lrepochdecay'))
    # Stacking layers into network
    input_layer = network = lasagne.layers.InputLayer(shape=(None, int(cp.get('NeuralNetwork','inputlayer'))),
                                        input_var=input_var)
    network = lasagne.layers.DropoutLayer(incoming=network,
                            p=float(cp.get('NeuralNetwork','corruptionfactor')))
    encoder_layer = network = lasagne.layers.DenseLayer(incoming=network,
                                          num_units=int(cp.get('NeuralNetwork','hiddenlayer')),
                                          W=lasagne.init.Normal(),
                                          nonlinearity=relu if enc_act == 'ReLU' else linear )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(cp.get('NeuralNetwork','outputlayer')),
                                        W=lasagne.init.Normal(),
                                        nonlinearity=relu if dec_act == 'ReLU' else linear )

    # Create train function
    learning_rate = T.scalar(name='learning_rate')
    prediction = lasagne.layers.get_output(network)
    cost = lasagne.objectives.squared_error(
        prediction, input_layer.input_var).mean()
    params = lasagne.layers.get_all_params(
        network, trainable=True)
    updates = lasagne.updates.momentum(
        cost, params, learning_rate=learning_rate)
    train = theano.function(
        inputs=[index, learning_rate], outputs=cost,
        updates=updates, givens={input_layer.input_var: input_var[index:index + batch_size, :]})

    lw_epochs = int(cp.get('NeuralNetwork','maxepochs'))
    base_lr = float(cp.get('NeuralNetwork','learningrate'))
    # Start training
    num = cp.get('Experiment','num')
    for epoch in xrange(lw_epochs):
        epoch_loss = 0
        for row in xrange(0, dataset.shape[0], batch_size):
            loss = train(row, base_lr)
            epoch_loss += loss
        epoch_loss = float(epoch_loss) / (dataset.shape[0]/batch_size)
        if epoch % 10 == 0:
            log(str(epoch) + ' ' + str(epoch_loss),
                label='LWT-Layer' + str(num))
        if (epoch % lr_decay == 0 and epoch != 0):
            base_lr = base_lr / 10
        if (epoch % 100 == 0) and (epoch != 0):
            utils.save('autoenc_'+num+'.zip',network)
    input_layer.input_var = input_var
    prefix = cp.get('Experiment','prefix')
    utils.save(prefix+'_autoencoder_'+num+'.zip',network)
    np.save(prefix+'_'+num+'_W1.npy',encoder_layer.W.eval())
    np.save(prefix+'_'+num+'_W2.npy',network.W.eval())
    np.save(prefix+'_'+num+'_b1.npy',encoder_layer.b.eval())
    np.save(prefix+'_'+num+'_b2.npy',network.b.eval())
    hidden = lasagne.layers.get_output(encoder_layer).eval()
    log(hidden.shape)
    np.save(prefix+'_hidden.npy',hidden)

def main(path):
    cp = load_config(path)
    try:
        [X,labels] = load_data(cp)
    except:
        X = load_data(cp)
    init(cp,X)


from operator import attrgetter
from argparse import ArgumentParser
if __name__ == "__main__":
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='config file')
    opts = parser.parse_args()
    getter = attrgetter('input')
    inp = getter(opts)
    main(inp)
