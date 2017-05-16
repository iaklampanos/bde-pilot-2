import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES'
#os.environ['THEANO_FLAGS'] = 'device=cpu,force_device=True'
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
    log('Loading data........')
    if cp.get('Experiment', 'inputfile') == '':
        [X1, labels1] = utils.load_mnist(
            dataset='training', path='/home/ubuntu/data/mnist/')
        [X2, labels2] = utils.load_mnist(
            dataset='testing', path='/home/ubuntu/data/mnist/')
        X = np.concatenate((X1, X2), axis=0)
        labels = np.concatenate((labels1, labels2), axis=0)
        # if train == 'train':
        X = X.astype(np.float32) * 0.02
        #     np.random.shuffle(X)
        return [X[0:10], labels]
    else:
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
        #     X[i, 3] = scipy.misc.imresize(X[i, 3], 167, 167))
        return X
    log('DONE........')

def make_weather(cp, dataset):
    # Init shallow weather nnet
    varidx = int(cp.get('Weather', 'varidx'))
    lvlidx = int(cp.get('Weather', 'lvlidx'))
    featurex = int(cp.get('Weather', 'feature_x'))
    featurey = int(cp.get('Weather', 'feature_y'))
    channels = int(cp.get('Weather', 'channels'))
    dataset = dataset[:, 4]
    dataset = [x for x in dataset]
    dataset = np.array(dataset)
    dataset = dataset[:, varidx, 0:lvlidx, :, :]
    dataset = dataset.reshape(dataset.shape[0],channels*featurex*featurey)
    dataset = minmax_scale(dataset)
    log(dataset.shape)
    featurex = int(cp.get('Weather', 'feature_x'))
    featurey = int(cp.get('Weather', 'feature_y'))
    channels = int(cp.get('Weather', 'channels'))
    dataset = dataset.reshape(dataset.shape[0], 1, channels, featurex, featurey)
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    return input_var


def init_weather_conv(cp, dataset):
    # Init shallow weather nnet
    featurex = int(cp.get('Weather', 'feature_x'))
    featurey = int(cp.get('Weather', 'feature_y'))
    varidx = int(cp.get('Weather', 'varidx'))
    lvlidx = int(cp.get('Weather', 'lvlidx'))
    channels = int(cp.get('Weather', 'channels'))
    dataset = dataset[:, 4]
    dataset = [x for x in dataset]
    dataset = np.array(dataset)
    dataset = dataset[:, varidx, 0:lvlidx, :, :]
    log(dataset.shape)
    dataset = dataset.reshape(dataset.shape[0],channels*featurex*featurey)
    dataset = minmax_scale(dataset)
    channels = int(cp.get('Weather', 'channels'))
    dataset = dataset.reshape(dataset.shape[0], 1, channels, featurex, featurey)
    log(dataset.shape)
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    def_filters = int(cp.get('Default','filters'))
    def_filter_size = int(cp.get('Default','filter_size'))
    def_stride = int(cp.get('Default','stride'))
    def_padding = int(cp.get('Default','padding'))
    channels = int(cp.get('Weather', 'channels'))
    pool_size = int(cp.get('W1', 'pool'))
    # Layer stacking
    input_layer = network = lasagne.layers.InputLayer(shape=(None, 1, channels, featurex, featurey),
                                                      input_var=input_var)
    network = lasagne.layers.Conv3DLayer(incoming=network,
                                         num_filters=def_filters, filter_size=def_filter_size,
                                         stride=def_stride,
                                         pad=def_padding
                                         )
    network = lasagne.layers.Conv3DLayer(incoming=network,
                                         num_filters=int(cp.get('W1', 'convfilters')),
                                          filter_size=(2,int(cp.get('W1', 'filtersize')),int(cp.get('W1', 'filtersize'))),
                                         stride=(1,int(cp.get('W1', 'stride')),int(cp.get('W1', 'stride'))),
                                         pad=(0,int(cp.get('W1','pad')),int(cp.get('W1','pad')))
                                         )
    network = lasagne.layers.MaxPool3DLayer(
        incoming=network, pool_size=(1,pool_size,pool_size))
    network2 = input_layer
    network2 = lasagne.layers.Conv3DLayer(incoming=network2,
                                         num_filters=def_filters, filter_size=def_filter_size,
                                         stride=def_stride,
                                         pad=def_padding
                                         )
    network2 = lasagne.layers.Conv3DLayer(incoming=network2,
                                         num_filters=int(cp.get('W2', 'convfilters')),
                                          filter_size=(2,int(cp.get('W2', 'filtersize')),int(cp.get('W2', 'filtersize'))),
                                         stride=(1,int(cp.get('W2', 'stride')),int(cp.get('W2', 'stride'))),
                                         pad=(0,int(cp.get('W2','pad')),int(cp.get('W2','pad')))
                                         )
    network2 = lasagne.layers.MaxPool3DLayer(
        incoming=network2, pool_size=(1,pool_size,pool_size))
    network = lasagne.layers.FlattenLayer(network)
    network2 = lasagne.layers.FlattenLayer(network2)
    out = lasagne.layers.ConcatLayer(incomings=(network,network2),axis=1)
    # log('Printing Weather Net Structure.......')
    # log(lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(network)))
    return [input_layer, input_var, out]


def make_disp(cp, dataset):
    # Init shallow dispersion nnet
    dataset = dataset[:, 3]
    featurex = int(cp.get('Dispersion', 'feature_x'))
    featurey = int(cp.get('Dispersion', 'feature_y'))
    channels = int(cp.get('Dispersion', 'channels'))
    dataset = [scipy.misc.imresize(x, (featurex, featurey)) for x in dataset]
    dataset = np.array(dataset)
    dataset = dataset.reshape(dataset.shape[0],featurex*featurey)
    dataset = maxabs_scale(dataset)
    dataset = dataset.reshape(dataset.shape[0], channels, featurex, featurey)
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    return input_var


def init_disp_conv(cp, dataset):
    # Init shallow dispersion nnet
    dataset = dataset[:, 3]
    featurex = int(cp.get('Dispersion', 'feature_x'))
    featurey = int(cp.get('Dispersion', 'feature_y'))
    channels = int(cp.get('Dispersion', 'channels'))
    pool_size = int(cp.get('D1', 'pool'))
    def_filters = int(cp.get('Default','filters'))
    def_filter_size = int(cp.get('Default','filter_size'))
    def_stride = int(cp.get('Default','stride'))
    def_padding = int(cp.get('Default','padding'))
    dataset = [scipy.misc.imresize(x, (featurex, featurey)) for x in dataset]
    dataset = np.array(dataset)
    dataset = dataset.reshape(dataset.shape[0],featurex*featurey)
    dataset = maxabs_scale(dataset)
    dataset = dataset.reshape(dataset.shape[0], channels, featurex, featurey)
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    # Layer stacking
    input_layer = network = lasagne.layers.InputLayer(shape=(None, channels, featurex, featurey),
                                                      input_var=input_var)
    pad = network = lasagne.layers.PadLayer(incoming=network,width=((0,1),(1,0)))
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=def_filters, filter_size=(
                                             def_filter_size, def_filter_size),
                                         stride=def_stride,
                                         pad=def_padding
                                         )
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=int(cp.get('D1', 'convfilters')), filter_size=(
                                             int(cp.get('D1', 'filtersize')), int(cp.get('D1', 'filtersize'))),
                                         stride=int(cp.get('D1', 'stride')),
                                         pad=int(cp.get('D1','pad')))
    network = lasagne.layers.MaxPool2DLayer(
        incoming=network, pool_size=(pool_size, pool_size))
    network2 = pad
    network2 = lasagne.layers.Conv2DLayer(incoming=network2,
                                         num_filters=def_filters, filter_size=(
                                             def_filter_size, def_filter_size),
                                         stride=def_stride,
                                         pad=def_padding
                                         )
    network2 = lasagne.layers.Conv2DLayer(incoming=network2,
                                         num_filters=int(cp.get('D2', 'convfilters')), filter_size=(
                                             int(cp.get('D2', 'filtersize')), int(cp.get('D2', 'filtersize'))),
                                         stride=int(cp.get('D2', 'stride')),
                                         pad=int(cp.get('D2','pad')))
    network2 = lasagne.layers.MaxPool2DLayer(
        incoming=network2, pool_size=(pool_size, pool_size))
    network = lasagne.layers.FlattenLayer(network)
    network2 = lasagne.layers.FlattenLayer(network2)
    out = lasagne.layers.ConcatLayer(incomings=(network,network2),axis=1)
    # log('Printing Dispersion Net Structure.......')
    # log(lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(network)))
    return [input_layer, input_var, out]


def init(cp, dataset):
    prefix = cp.get('Experiment','prefix')
    output = cp.get('Experiment','output')
    [win_layer, win, weather_net] = init_weather_conv(cp, dataset)
    [din_layer, din, disp_net] = init_disp_conv(cp, dataset)
    batch_size = int(cp.get('NeuralNetwork', 'batchsize'))
    index = T.lscalar()
    lw_epochs = int(cp.get('NeuralNetwork', 'maxepochs'))
    base_lr = float(cp.get('NeuralNetwork', 'learningrate'))
    concat = network = lasagne.layers.ConcatLayer(
        incomings=(weather_net, disp_net), axis=1)
    network = lasagne.layers.DropoutLayer(incoming=network,
                                          p=float(cp.get('NeuralNetwork', 'corruptionfactor')))
    test = network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden0')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden1')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden2')),
                                        )
    # Softmax
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'stationnum')),
                                        nonlinearity=lasagne.nonlinearities.softmax
                                        )
    # log(lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(network)))
    from draw_net import draw_to_file
    layers = lasagne.layers.get_all_layers(network)
    draw_to_file(layers, output+prefix+'_network.pdf', output_shape=True,verbose=True)
    try:
        params = np.load(output+'sharing_model.npy')
        lasagne.layers.set_all_param_values(network,params)
        #test_w = test.W.eval()
        # log(str(np.array_equal(test_w,params[8])))
        log('Found pretrained model.....')
        log('Training with pretrained weights......')
        #log('Are weights equal? '+str(np.array_equal(test_w,params[8])))
    except:
        pass
    lr_decay = int(cp.get('NeuralNetwork','lrdecayepoch'))
    dist_var = T.fmatrix('targets')
    learning_rate = T.scalar(name='learning_rate')
    prediction = lasagne.layers.get_output(network)
    cost = lasagne.objectives.categorical_crossentropy(
        prediction, dist_var).mean()
    # acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
    #               dtype=theano.config.floatX)
    params = lasagne.layers.get_all_params(
        network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        cost, params, learning_rate=learning_rate, momentum=0.9)
    train = theano.function(
        inputs=[index, dist_var, learning_rate], outputs=cost,
        updates=updates, givens={win_layer.input_var: win[index:index + batch_size, :],
                             din_layer.input_var: din[index:index + batch_size, :]})
    try:
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
                    label='Supervised')
            if (epoch % lr_decay == 0 ) and (epoch != 0):
                base_lr = base_lr / 10.0
            if (epoch % 50 == 0) and (epoch != 0) :
                np.save(output+prefix + '_model.npy',lasagne.layers.get_all_param_values(network))
        log('Saving......')
        np.save(output+prefix + '_model.npy',lasagne.layers.get_all_param_values(network))
        np.save(output+'sharing_model.npy',lasagne.layers.get_all_param_values(network))
    except KeyboardInterrupt:
        #log('Saving......')
        np.save(output+prefix + '_model.npy',lasagne.layers.get_all_param_values(network))
        np.save(output+'sharing_model.npy',lasagne.layers.get_all_param_values(network))
    # win_layer.input_var = make_weather(cp,dataset_test)
    # din_layer.input_var = make_disp(cp,dataset_test)
    # prediction = lasagne.layers.get_output(network).argmax(axis=1).eval()
    # print prediction[0:40]
    # print dataset_test[:,1][0:40]
    # acc = np.mean((prediction==dataset_test[:,1]))
    # print prediction.shape
    # log('ACC:  '+str(acc))
    # print pred
    # acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
    #               dtype=theano.config.floatX)


def init_pretrained(cp, dataset_test, batch):
    output = cp.get('Experiment','output')
    prefix = cp.get('Experiment','prefix')
    [win_layer, win, weather_net] = init_weather_conv(cp, dataset_test)
    [din_layer, din, disp_net] = init_disp_conv(cp, dataset_test)
    batch_size = int(cp.get('NeuralNetwork', 'batchsize'))
    index = T.lscalar()
    lw_epochs = int(cp.get('NeuralNetwork', 'maxepochs'))
    base_lr = float(cp.get('NeuralNetwork', 'learningrate'))
    concat = network = lasagne.layers.ConcatLayer(
        incomings=(weather_net, disp_net), axis=1)
    network = lasagne.layers.DropoutLayer(incoming=network,
                                          p=float(cp.get('NeuralNetwork', 'corruptionfactor')))
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden0')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden1')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden2')),
                                        )
    # Softmax
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'stationnum')),
                                        nonlinearity=lasagne.nonlinearities.softmax
                                        )
    log(lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(network)))
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
    np.save(output+prefix+'_test_results_'+str(batch)+'.npy',results)



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
        # X_test = np.load('super_test.npy')
        np.random.shuffle(X)
        # X = X[0:100,:]
        #X_test = X_test[0:100,:]
        # print X_test.shape
        for i in xrange(0, X.shape[0], 3000):
            init_pretrained(cp, X[i:i+3000,:], i)


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
