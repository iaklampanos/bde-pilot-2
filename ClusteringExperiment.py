import dataset_utils as utils
from Dataset_transformations import Dataset_transformations
import numpy as np

class ClusteringExperiment(object):

     def __init__(self,train_dataset,nnet,clust_obj,test_dataset=None):
         self._nnet = nnet
         self._clustering = clust_obj
         self._training_dataset = train_dataset
         print self._training_dataset.get_items().shape
         if not(test_dataset is None):
             self._test_dataset = test_dataset
         if self._clustering._features_first:
             self._training_dataset._items = np.transpose(self._training_dataset.get_items())
             print self._training_dataset.get_items().shape
             if not(test_dataset is None):
                 self._test_dataset._items = np.transpose(self._test_dataset.get_items())

     def start(self):
         data = self._training_dataset.get_items()
         print data.shape
         self._nnet.train()
         encodings = self._nnet.hidden
         output = self._nnet.decoded
         self._encoding = Dataset_transformations(encodings,self._nnet.hidden_size/100,encodings.shape)
         self._output = Dataset_transformations(output,self._nnet.m/100,output.shape)

     def test(self,data):
        self._nnet.test(data)
        encodings = self._nnet.get_hidden()
        output = self._nnet.get_output()
        self._encoding = Dataset_transformations(encodings,self._nnet.hidden_size/100,encodings.shape)
        self._output = Dataset_transformations(output,self._nnet.m/100,output.shape)

     def clustering(self):
         self._clustering._items = self._encoding.get_items()
         print self._clustering.get_items().shape
         self._clustering.kmeans()

     def plot_output_frames(self,x,y,iter_=40):
         inp = self._training_dataset.get_items()
         output = self._output.get_items()
         print inp.shape
         print output.shape
         for i in range(0,iter_):
             utils.plot_pixel_image(inp[i],output[i],x,y)


     def save(self, filename='Clustering_experiment.zip'):
         utils.save(filename, self)

     def load(self, filename='Clustering_experiment.zip'):
         self = utils.load(filename)
