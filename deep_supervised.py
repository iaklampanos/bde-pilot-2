import os
os.environ[
    'THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES'
# os.environ['THEANO_FLAGS'] = 'device=cpu,force_device=True'
import sys
import ConfigParser
import numpy as np
import theano
from theano import tensor as T
import datetime
from datetime import datetime
import dataset_utils as utils
import lasagne
from Unpool2DLayer import Unpool2DLayer
import dataset_utils as utils
from sklearn.metrics import accuracy_score
from lasagne.regularization import regularize_layer_params, l2
from modeltemplate import Model
import scipy.misc
from sklearn.preprocessing import maxabs_scale,minmax_scale




def load_config(input_path):
    cp = ConfigParser.ConfigParser()
    cp.read(input_path)
    return cp


def log(s, label='INFO'):
    sys.stdout.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')
    sys.stdout.flush()


def load_data(cp, train):
    try:
        X = np.load(cp.get('Experiment', 'inputfile'))
        try:
           X = X[cp.get('Experiment','label')]
        except:
           pass
        # if train == 'train':
        print np.unique(X[:,1])
        p = np.random.permutation(X.shape[0])
        X = X[p]
        prefix = cp.get('Experiment', 'prefix')
        output = cp.get('Experiment','output')
        np.save(output+prefix + '_random_perm.npy', p)
        #  for i in xrange(X.shape[0]):
        #     X[i, 3] = scipy.misc.imresize(X[i, 3], (167, 167))
        return X[0:500,:]
        log('DONE........')
    except:
        raise ValueError('Fill in the inputfile on convdeep.ini configuration file')

def make_weather(cp, dataset):
    w_x = int(cp.get('Weather','feature_x'))
    w_y = int(cp.get('Weather','feature_y'))
    channels = int(cp.get('Weather','channels'))
    varidx = int(cp.get('Weather', 'varidx'))
    lvlidx = int(cp.get('Weather', 'lvlidx'))
    local_dat = dataset[:, 4]
    local_dat = [x for x in local_dat]
    local_dat = np.array(local_dat)
    local_dat = local_dat[:, varidx, lvlidx, :, :]
    try:
        norm = cp.get('Experiment','normalize')
        log('Performing minmax scale....')
        local_dat = minmax_scale(local_dat)
    except:
        pass
    local_dat = local_dat.reshape(local_dat.shape[0],w_x*w_y)
    local_dat = local_dat.reshape(local_dat.shape[0], channels, w_y, w_y)
    input_var = theano.shared(name='input_var', value=np.asarray(local_dat,
                                             dtype=theano.config.floatX),
                              borrow=True)
    return input_var




def make_disp(cp, dataset):
    local_dat = dataset[:, 3]
    d_x = int(cp.get('Dispersion', 'feature_x'))
    d_y = int(cp.get('Dispersion', 'feature_y'))
    channels = int(cp.get('Dispersion', 'channels'))
    try:
        norm = cp.get('Experiment','resize')
        log('Performing image resize....')
        local_dat = [scipy.misc.imresize(x, (d_x, d_y)) for x in local_dat]
        local_dat = maxabs_scale(local_dat)
    except:
        pass
    local_dat = [x for x in local_dat]
    local_dat = np.array(local_dat)
    local_dat = local_dat.reshape(local_dat.shape[0],d_x*d_y)
    local_dat = local_dat.reshape(local_dat.shape[0], channels, d_x, d_y)
    input_var = theano.shared(name='input_var', value=np.asarray(local_dat,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    return input_var

def get_default(cp, network):
    filters = int(cp.get('Default','filters'))
    filter_size = int(cp.get('Default','filter_size'))
    stride = int(cp.get('Default','stride'))
    padding = int(cp.get('Default','padding'))
    return lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=filters, filter_size=(
                                             filter_size, filter_size),
                                         stride=stride,
                                         pad=padding)

### Weather-wise first "umbrella" layer
def W1(cp, network):
    pool_size = int(cp.get('W1','pool'))
    network = get_default(cp, network)
    return lasagne.layers.MaxPool2DLayer(incoming=network,pool_size=(pool_size, pool_size))

### Weather-wise second "umbrella" layer
def W2(cp, network):
    pool_size = int(cp.get('W2','pool'))
    filters = int(cp.get('W2','filters'))
    filter_size = int(cp.get('W2','filter_size'))
    stride = int(cp.get('W2','stride'))
    padding = int(cp.get('W2','padding'))
    network = get_default(cp, network)
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=filters, filter_size=(
                                             filter_size, filter_size),
                                         stride=stride,
                                         pad=padding)
    return lasagne.layers.MaxPool2DLayer(incoming=network,pool_size=(pool_size, pool_size))


### Weather-wise third "umbrella" layer
def W3(cp, network):
    pool_size = int(cp.get('W3','pool'))
    filters = int(cp.get('W3','filters'))
    filter_size = int(cp.get('W3','filter_size'))
    stride = int(cp.get('W3','stride'))
    padding = int(cp.get('W3','padding'))
    network = get_default(cp, network)
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=filters, filter_size=(
                                             filter_size, filter_size),
                                         stride=stride,
                                         pad=padding)
    return lasagne.layers.MaxPool2DLayer(incoming=network,pool_size=(pool_size, pool_size))

def W4(cp, network):
    pool_size1 = int(cp.get('W4','pool1'))
    pool_size2 = int(cp.get('W4','pool2'))
    network = lasagne.layers.MaxPool2DLayer(incoming=network,pool_size=(pool_size1, pool_size1))
    network = get_default(cp, network)
    return lasagne.layers.MaxPool2DLayer(incoming=network,pool_size=(pool_size2, pool_size2))


### Weather wise layer
def weather_net(cp, dataset):
    w_x = int(cp.get('Weather','feature_x'))
    w_y = int(cp.get('Weather','feature_y'))
    channels = int(cp.get('Weather','channels'))
    varidx = int(cp.get('Weather', 'varidx'))
    lvlidx = int(cp.get('Weather', 'lvlidx'))
    local_dat = dataset[:, 4]
    local_dat = [x for x in local_dat]
    local_dat = np.array(local_dat)
    local_dat = local_dat[:, varidx, lvlidx, :, :]
    local_dat = local_dat.reshape(local_dat.shape[0],w_x*w_y)
    try:
        norm = cp.get('Experiment','normalize')
        log('Performing minmax scale....')
        local_dat = minmax_scale(local_dat)
    except:
        pass
    local_dat = local_dat.reshape(local_dat.shape[0], channels, w_y, w_y)
    # Convert dataset into symbolic variable
    input_var = theano.shared(name='input_var', value=np.asarray(local_dat,
                                             dtype=theano.config.floatX),
                              borrow=True)
    input_layer = lasagne.layers.InputLayer(shape=(None, channels, w_x, w_y),
                                                      input_var=input_var)
    w1 = W1(cp,input_layer)
    w2 = W2(cp,input_layer)
    w3 = W3(cp,input_layer)
    w4 = W4(cp,input_layer)
    w1 = lasagne.layers.FlattenLayer(incoming=w1)
    w2 = lasagne.layers.FlattenLayer(incoming=w2)
    w3 = lasagne.layers.FlattenLayer(incoming=w3)
    w4 = lasagne.layers.FlattenLayer(incoming=w4)
    # for w in [w1,w2,w3,w4]:
    #     print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    #     print ' '
    #     print lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(w))
    #     print ' '
    #     print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    return [input_layer, input_var, lasagne.layers.ConcatLayer(incomings=(w1,w2,w3,w4),axis=1)]


def D1(cp, network):
    pool_size1 = int(cp.get('D1','pool1'))
    pool_size2 = int(cp.get('D1','pool2'))
    network = get_default(cp, network)
    network = lasagne.layers.MaxPool2DLayer(incoming=network,pool_size=(pool_size1, pool_size1))
    return lasagne.layers.MaxPool2DLayer(incoming=network,pool_size=(pool_size2, pool_size2))

def D2(cp, network):
    pool_size = int(cp.get('D2','pool'))
    filters = int(cp.get('D2','filters'))
    filter_size = int(cp.get('D2','filter_size'))
    stride = int(cp.get('D2','stride'))
    padding = int(cp.get('D2','padding'))
    network = get_default(cp, network)
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=filters, filter_size=(
                                             filter_size, filter_size),
                                         stride=stride,
                                         pad=padding)
    network = lasagne.layers.PadLayer(incoming=network,width=((0,3),(3,0)))
    return lasagne.layers.MaxPool2DLayer(incoming=network,pool_size=(pool_size, pool_size))


def D3(cp, network):
    pool_size = int(cp.get('D3','pool'))
    filters = int(cp.get('D3','filters'))
    filter_size = int(cp.get('D3','filter_size'))
    stride = int(cp.get('D3','stride'))
    padding = int(cp.get('D3','padding'))
    network = get_default(cp, network)
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=filters, filter_size=(
                                             filter_size, filter_size),
                                         stride=stride,
                                         pad=padding)
    network = lasagne.layers.PadLayer(incoming=network,width=((0,3),(3,0)))
    return lasagne.layers.MaxPool2DLayer(incoming=network,pool_size=(pool_size, pool_size))

def D4(cp, network):
    pool_size1 = int(cp.get('D1','pool1'))
    pool_size2 = int(cp.get('D1','pool2'))
    network = lasagne.layers.MaxPool2DLayer(incoming=network,pool_size=(pool_size1, pool_size1))
    network = get_default(cp, network)
    return lasagne.layers.MaxPool2DLayer(incoming=network,pool_size=(pool_size2, pool_size2))



def dispersion_net(cp, dataset):
    local_dat = dataset[:, 3]
    d_x = int(cp.get('Dispersion', 'feature_x'))
    d_y = int(cp.get('Dispersion', 'feature_y'))
    channels = int(cp.get('Dispersion', 'channels'))
    try:
        norm = cp.get('Experiment','resize')
        log('Performing image resize....')
        local_dat = [scipy.misc.imresize(x, (d_x, d_y)) for x in local_dat]
        local_dat = maxabs_scale(local_dat)
    except:
        pass
    local_dat = [x for x in local_dat]
    local_dat = np.array(local_dat)
    local_dat = local_dat.reshape(local_dat.shape[0],d_x*d_y)
    local_dat = local_dat.reshape(local_dat.shape[0], channels, d_x, d_y)
    input_var = theano.shared(name='input_var', value=np.asarray(local_dat,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    input_layer = network = lasagne.layers.InputLayer(shape=(None, channels, d_x, d_y),
                                                      input_var=input_var)
    network = lasagne.layers.PadLayer(incoming=network,width=((0,1),(1,0)))
    d1 = D1(cp,network)
    d2 = D2(cp,network)
    d3 = D3(cp,network)
    d4 = D4(cp,network)
    d1 = lasagne.layers.FlattenLayer(incoming=d1)
    d2 = lasagne.layers.FlattenLayer(incoming=d2)
    d3 = lasagne.layers.FlattenLayer(incoming=d3)
    d4 = lasagne.layers.FlattenLayer(incoming=d4)
    # for d in [d1,d2,d3,d4]:
    #     print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    #     print ' '
    #     print lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(d))
    #     print ' '
    #     print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    return [input_layer, input_var, lasagne.layers.ConcatLayer(incomings=(d1,d2,d3,d4),axis=1)]


def Deep1(cp, network):
    pool_size = cp.getint('Con1','pool')
    network = get_default(cp, network)
    return lasagne.layers.MaxPool2DLayer(incoming=network,pool_size=(pool_size, pool_size))

def Deep2(cp, network):
    pool_size = int(cp.get('Con2','pool'))
    filters = int(cp.get('Con2','filters'))
    filter_size = int(cp.get('Con2','filter_size'))
    stride = int(cp.get('Con2','stride'))
    padding = int(cp.get('Con2','padding'))
    network = get_default(cp, network)
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=filters, filter_size=(
                                             filter_size, filter_size),
                                         stride=stride,
                                         pad=padding)
    return lasagne.layers.MaxPool2DLayer(incoming=network,pool_size=(pool_size, pool_size))


def Deep3(cp, network):
    pool_size = int(cp.get('Con3','pool'))
    filters = int(cp.get('Con3','filters'))
    filter_size = int(cp.get('Con3','filter_size'))
    stride = int(cp.get('Con3','stride'))
    padding = int(cp.get('Con3','padding'))
    network = get_default(cp, network)
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=filters, filter_size=(
                                             filter_size, filter_size),
                                         stride=stride,
                                         pad=padding)
    return lasagne.layers.MaxPool2DLayer(incoming=network,pool_size=(pool_size, pool_size))

def Deep4(cp, network):
    pool_size = cp.getint('Con1','pool')
    network = lasagne.layers.MaxPool2DLayer(incoming=network,pool_size=(pool_size, pool_size))
    return get_default(cp, network)


def init(cp, dataset):
    channels = cp.getint('NeuralNetwork','channels')
    deep_x = cp.getint('NeuralNetwork','feature_x')
    deep_y = cp.getint('NeuralNetwork','feature_y')
    prefix = cp.get('Experiment','prefix')
    output = cp.get('Experiment','output')
    batch_size = int(cp.get('NeuralNetwork', 'batchsize'))
    index = T.lscalar()
    lw_epochs = int(cp.get('NeuralNetwork', 'maxepochs'))
    base_lr = float(cp.get('NeuralNetwork', 'learningrate'))
    [win_layer, win, w_net] = weather_net(cp, dataset)
    [din_layer, din, d_net] = dispersion_net(cp, dataset)
    # print lasagne.layers.get_output_shape(w_net)
    # print lasagne.layers.get_output_shape(d_net)
    deep_net = lasagne.layers.ConcatLayer(incomings=(w_net,d_net),axis=1)
    deep_net = lasagne.layers.DenseLayer(incoming=deep_net,
                                         num_units=int(cp.get('NeuralNetwork','Condense0'))
                                         )
    deep_net = lasagne.layers.DenseLayer(incoming=deep_net,
                                         num_units=int(cp.get('NeuralNetwork','Condense1'))
                                         )
    deep_net = lasagne.layers.ReshapeLayer(incoming=deep_net,shape=([0],
                                                       channels,deep_x,deep_y))

    deep1 = Deep1(cp, deep_net)
    deep2 = Deep2(cp, deep_net)
    deep3 = Deep3(cp, deep_net)
    deep4 = Deep4(cp, deep_net)
    deep1 = lasagne.layers.FlattenLayer(incoming=deep1)
    deep2 = lasagne.layers.FlattenLayer(incoming=deep2)
    deep3 = lasagne.layers.FlattenLayer(incoming=deep3)
    deep4 = lasagne.layers.FlattenLayer(incoming=deep4)
    # for d in [deep1,deep2,deep3,deep4]:
    #     print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    #     print ' '
    #     print lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(d))
    #     print ' '
    #     print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'

    deep_net = lasagne.layers.ConcatLayer(
        incomings=(deep1,deep2,deep3,deep4), axis=1)
    deep_net = lasagne.layers.DropoutLayer(incoming=deep_net,p=cp.getfloat('NeuralNetwork','corruptionfactor'))
    deep_net = lasagne.layers.DenseLayer(incoming=deep_net,
                                         num_units=int(cp.get('NeuralNetwork','Condense2'))
                                         )
    deep_net = lasagne.layers.DenseLayer(incoming=deep_net,
                                         num_units=int(cp.get('NeuralNetwork','Condense3'))
                                         )
    deep_net = lasagne.layers.DenseLayer(incoming=deep_net,
                                         num_units=int(cp.get('NeuralNetwork','Condense4'))
                                         )
    # # Softmax
    deep_net = lasagne.layers.DenseLayer(incoming=deep_net,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'stationnum')),
                                        nonlinearity=lasagne.nonlinearities.softmax
                                        )

    # log(lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(deep_net)))

    try:
        params = np.load(output+'sharing_model.npy')
        lasagne.layers.set_all_param_values(deep_net,params)
        test_w = test.W.eval()
        # log(str(np.array_equal(test_w,params[8])))
        log('Found pretrained model.....')
        log('Training with pretrained weights......')
    except:
        pass
    lr_decay = int(cp.get('NeuralNetwork','lrdecayepoch'))
    dist_var = T.fmatrix('targets')
    learning_rate = T.scalar(name='learning_rate')
    prediction = lasagne.layers.get_output(deep_net)
    cost = lasagne.objectives.categorical_crossentropy(
        prediction, dist_var).mean()
    # acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
    #               dtype=theano.config.floatX)
    params = lasagne.layers.get_all_params(
        deep_net, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        cost, params, learning_rate=learning_rate, momentum=0.9)
    train = theano.function(
        inputs=[index, dist_var, learning_rate], outputs=cost,
        updates=updates, givens={win_layer.input_var: win[index:index + batch_size, :],
                             din_layer.input_var: din[index:index + batch_size, :]})

    for epoch in xrange(lw_epochs):
        epoch_loss = 0
        for row in xrange(0, dataset.shape[0], batch_size):
            pos = dataset[row:row+batch_size,1]
            x = np.zeros(shape=(len(pos),20),dtype=np.float32)
            for i,arr in enumerate(x):
                arr[pos[i]] = 1
            loss = train(row, x, base_lr)
            epoch_loss += loss
        epoch_loss = float(epoch_loss) / (dataset.shape[0] * 1.0 / batch_size)
        if epoch % 10 == 0:
            log(str(epoch) + ' ' + str(epoch_loss),
                label='Deep-Supervised')
        if (epoch % lr_decay == 0 ) and (epoch != 0):
            base_lr = base_lr / 10.0
        if (epoch % 100 == 0) and (epoch != 0) :
            np.save(output+prefix + '_model.npy',lasagne.layers.get_all_param_values(deep_net))
    log('Saving......')


def init_pretrained(cp, dataset_test):
    channels = cp.getint('NeuralNetwork','channels')
    deep_x = cp.getint('NeuralNetwork','feature_x')
    deep_y = cp.getint('NeuralNetwork','feature_y')
    prefix = cp.get('Experiment','prefix')
    output = cp.get('Experiment','output')
    batch_size = int(cp.get('NeuralNetwork', 'batchsize'))
    index = T.lscalar()
    lw_epochs = int(cp.get('NeuralNetwork', 'maxepochs'))
    base_lr = float(cp.get('NeuralNetwork', 'learningrate'))
    [win_layer, win, w_net] = weather_net(cp, dataset)
    [din_layer, din, d_net] = dispersion_net(cp, dataset)
    # print lasagne.layers.get_output_shape(w_net)
    # print lasagne.layers.get_output_shape(d_net)
    deep_net = lasagne.layers.ConcatLayer(incomings=(w_net,d_net),axis=1)
    deep_net = lasagne.layers.DenseLayer(incoming=deep_net,
                                         num_units=int(cp.get('NeuralNetwork','Condense0'))
                                         )
    deep_net = lasagne.layers.DenseLayer(incoming=deep_net,
                                         num_units=int(cp.get('NeuralNetwork','Condense1'))
                                         )
    deep_net = lasagne.layers.ReshapeLayer(incoming=deep_net,shape=([0],
                                                       channels,deep_x,deep_y))

    deep1 = Deep1(cp, deep_net)
    deep2 = Deep2(cp, deep_net)
    deep3 = Deep3(cp, deep_net)
    deep4 = Deep4(cp, deep_net)
    deep1 = lasagne.layers.FlattenLayer(incoming=deep1)
    deep2 = lasagne.layers.FlattenLayer(incoming=deep2)
    deep3 = lasagne.layers.FlattenLayer(incoming=deep3)
    deep4 = lasagne.layers.FlattenLayer(incoming=deep4)
    # for d in [deep1,deep2,deep3,deep4]:
    #     print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    #     print ' '
    #     print lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(d))
    #     print ' '
    #     print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'

    deep_net = lasagne.layers.ConcatLayer(
        incomings=(deep1,deep2,deep3,deep4), axis=1)
    deep_net = lasagne.layers.DropoutLayer(incoming=deep_net,p=cp.getfloat('NeuralNetwork','corruptionfactor'))
    deep_net = lasagne.layers.DenseLayer(incoming=deep_net,
                                         num_units=int(cp.get('NeuralNetwork','Condense2'))
                                         )
    deep_net = lasagne.layers.DenseLayer(incoming=deep_net,
                                         num_units=int(cp.get('NeuralNetwork','Condense3'))
                                         )
    deep_net = lasagne.layers.DenseLayer(incoming=deep_net,
                                         num_units=int(cp.get('NeuralNetwork','Condense4'))
                                         )
    # # Softmax
    deep_net = lasagne.layers.DenseLayer(incoming=deep_net,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'stationnum')),
                                        nonlinearity=lasagne.nonlinearities.softmax
                                        )
    params = np.load(output+prefix+'_model.npy')
    print params.shape
    lasagne.layers.set_all_param_values(network,params)
    #model = Model(input_layer=[win_layer,din_layer],encoder_layer=None,decoder_layer=network,network=network)
    #model.save(output+prefix+'_model.zip')
    win_layer.input_var = make_weather(cp,dataset_test)
    din_layer.input_var = make_disp(cp,dataset_test)
    prediction = lasagne.layers.get_output(network).eval()
    max_pred = lasagne.layers.get_output(network).argmax(axis=1).eval()
    print max_pred[0:40]
    print dataset_test[:,1][0:40]
    acc = np.mean((max_pred==dataset_test[:,1]))
    print prediction.shape
    log('ACC:  '+str(acc))
    results = []
    for i in xrange(dataset_test.shape[0]):
        origin = dataset_test[i,1]
        raw_preds = prediction[i,:]
        scores = [(stat,pred) for stat,pred in enumerate(raw_preds)]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        results.append((origin,raw_preds,scores))
    results = np.asarray(results,dtype=object)
    np.save(output+prefix+'_test_results.npy',results)


def main(path, train):
    cp = load_config(path)
    # try:
    #     [X, labels] = load_data(cp, train)
    # except:
    X = load_data(cp, train)
    # init(cp, X_train, X_test)
    if train == 'train':
        init(cp, X)
    else:
        #X_test = np.load('super_test.npy')
        np.random.shuffle(X)
        #X_test = X_test[0:100,:]
        # print X_test.shape
        init_pretrained(cp, X)


from operator import attrgetter
from argparse import ArgumentParser
if __name__ == "__main__":
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='config file')
    parser.add_argument('-t', '--train', required=True, type=str,
                        help='training/testing')
    opts = parser.parse_args()
    getter = attrgetter('input', 'train')
    inp, train = getter(opts)
    main(inp, train)
