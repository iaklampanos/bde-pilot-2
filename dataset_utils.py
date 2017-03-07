import cPickle
import gzip
from netcdf_subset import netCDF_subset
import oct2py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
from netCDF4 import Dataset


def save(filename, *objects):
    fil = gzip.open(filename, 'wb')
    for obj in objects:
        cPickle.dump(obj, fil)
    fil.close()


def load(filename):
    fil = gzip.open(filename, 'rb')
    while True:
        try:
            yield cPickle.load(fil)
        except EOFError:
            break
    fil.close()


def load_single(filename):
    fil = gzip.open(filename, 'rb')
    c = cPickle.load(fil)
    fil.close()
    return c


def export_timebars(outp, start_date, nc_sub, clust_obj):
    oc = oct2py.Oct2Py()
    time_diagram = nc_sub.get_time_diagram(
        start_date, clust_obj._index_list)
    for n_clust, index in enumerate(time_diagram):
        oc.push('x', index)
        a = oc.eval('x*4;')
        oc.push('a', a)
        a = oc.eval('a+1;')
        oc.push('a', a)
        y = oc.zeros(1, clust_obj.get_items().shape[0])
        oc.push('y', y)
        oc.eval('y(a)=1;')
        oc.eval('bar(y)',
                plot_dir=outp, plot_name='cluster' + str(int(n_clust)) + '_timebar', plot_format='jpeg',
                plot_width='2048', plot_height='1536')


def plot_pixel_image(image, image2, x, y):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    pixels = image.reshape((x, y))
    ax.matshow(pixels, cmap=matplotlib.cm.binary)
    ax2 = fig.add_subplot(1, 2, 2)
    pixels2 = image2.reshape((x, y))
    ax2.matshow(pixels2, cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.plot()
    plt.show()


def plot_concentration(pollutant_array, x=45, y=150):
    fit = plt.figure()
    integ = np.zeros(shape=(pollutant_array.shape[
                     1] * pollutant_array.shape[2] * pollutant_array.shape[3]))
    for i in range(0,pollutant_array.shape[0]):
        integ += pollutant_array[i,0,:,:].flatten()
    print np.max(integ)
    integ = integ.reshape(pollutant_array.shape[2],pollutant_array.shape[3])
    integ_plot = integ[range(x,y),:]
    integ_plot = integ_plot[:,range(x,y)]
    plt.imshow(integ_plot,interpolation='nearest')
    plt.show()

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows * cols), dtype=float)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1)
                              * rows * cols]).reshape((rows * cols))
        labels[i] = lbl[ind[i]]
    for i in range(len(ind)):
        images[i] = images[i] / 255
    return images, labels


def export_descriptor_kmeans(outp, nc_sub, clust_obj):
    descriptors = clust_obj._descriptors
    for pos, desc in enumerate(descriptors):
        nc_sub.exact_copy_kmeans(
            outp + '/desc_kmeans_' + str(pos) + '.nc', desc)


def rename_descriptors(path):
    filelist = sorted(os.listdir(path))
    start_dts = [
        Dataset(path + '/' + f, 'r').SIMULATION_START_DATE for f in filelist]
    for pos, f in enumerate(filelist):
        os.rename(path + '/' + f, path + '/' +
                  f + '_' + start_dts[pos] + '.nc')


def export_descriptor_max(out_path, nc_sub, clust_obj):
    max_ret_list = nc_sub.find_continuous_timeslots(clust_obj._index_list)
    for pos, c in enumerate(max_ret_list):
        nc_sub.write_timetofile(out_path + '/cluster_descriptor_meanmax'
                                + str(pos) + '.nc', self._sub_pos,
                                range(c[0], c[1]), c_desc=True)


def export_middle_cluster_tofile(out_path,  nc_sub, clust_obj, desc_frames):
    max_ret_list = nc_sub.find_continuous_timeslots(clust_obj._index_list)
    print max_ret_list
    for pos, c in enumerate(max_ret_list):
        mid_start_plus = (c[len(c) - 1] - c[0] + 1) / \
            2 - desc_frames / 2
        mid_start = c[0] + mid_start_plus
        nc_sub.exact_copy_file(out_path +
                               '/cluster_descriptor' +
                               str(pos) + '.nc',
                               range(mid_start, mid_start + desc_frames))


def export_cluster_descriptor_middle(out_path,  nc_sub, clust_obj, desc_frames, desc_div):
    max_ret_list = nc_sub.find_continuous_timeslots(clust_obj._index_list)
    for pos, c in enumerate(max_ret_list):
        if (c[1] - c[0]) >= desc_frames:
            nc_sub.exact_copy_mean(out_path +
                                   '/cluster_descriptor_meanmiddle'
                                   + str(pos) + '.nc', range(c[0], c[0] + desc_frames), desc_frames, desc_div)
        else:
            nc_sub.exact_copy_mean(out_path +
                                   '/cluster_descriptor_meanmiddle' + str(pos) +
                                   '.nc', range(c[0], c[1]), len(range(c[0], c[1])), desc_div)
