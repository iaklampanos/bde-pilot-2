from operator import attrgetter
from argparse import ArgumentParser
import numpy as np
import os
from sklearn import metrics
from sklearn.cluster import KMeans,AgglomerativeClustering
from theano_autoencoder2 import AutoEncoder,load_mnist
import pickle


if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-ip', '--input', required=True, type=str,
                        help='input path')
    parser.add_argument('-il', '--layer', type=str,
                        help='input layer')
    opts = parser.parse_args()
    getter = attrgetter('input', 'layer')
    inp, layer = getter(opts)
    f_names = ['kmeans','10','100','100_10','100_100','500_10','500_100','500_2000_10',
                 '500_2000_100','500_500_2000_10']
    last_nmi = []
    last_ri = []
    if layer == 'hidden':
        for f in f_names:
            os.chdir(inp+'/'+f)
            nmi = np.load('nmi.npy')
            RI = np.load('RI.npy')
            last_nmi.append(nmi[15])
            last_ri.append(RI[15])
        os.chdir(inp)
        np.save('eval_ri_15.npy',last_ri)
        np.save('eval_nmi_15.npy',last_nmi)
    else:
        [X,labels] = load_mnist()
        labels = labels.flatten()
        for pos,f in enumerate(f_names):
            if f == 'kmeans':
                os.chdir(inp+'/'+f)
                nmi = np.load('nmi.npy')
                RI = np.load('RI.npy')
                last_nmi.append(nmi[15])
                last_ri.append(RI[15])
            elif pos>=1 and pos<=2:
                os.chdir(inp+'/'+f)
                print pos
                if pos == 1:
                    A1 = pickle.load(open('autoencoders_10.pkl','rb'))
                else:
                    A1 = pickle.load(open('autoencoders_100.pkl','rb'))
                feat = A1.get_hidden(X)
                V = AgglomerativeClustering(n_clusters=15,
                            affinity='cosine', linkage='average').fit(feat).labels_
                last_nmi.append(metrics.normalized_mutual_info_score(labels,V))
                last_ri.append(metrics.adjusted_rand_score(labels,V))
            elif pos>=3 and pos<=6:
                os.chdir(inp+'/'+f)
                print pos
                if pos == 3:
                    A1 = pickle.load(open('autoencoders_100.pkl','rb'))
                    A2 = pickle.load(open('autoencoders_10.pkl','rb'))
                elif pos == 4:
                    A1 = pickle.load(open('autoencoders_100.pkl','rb'))
                    A2 = pickle.load(open('autoencoders_1002.pkl','rb'))
                elif pos == 5:
                    A1 = pickle.load(open('autoencoders_500.pkl','rb'))
                    A2 = pickle.load(open('autoencoders_10.pkl','rb'))
                elif pos == 6:
                    A1 = pickle.load(open('autoencoders_500.pkl','rb'))
                    A2 = pickle.load(open('autoencoders_100.pkl','rb'))
                feat = A2.get_output(A1.get_hidden(X))
                V = AgglomerativeClustering(n_clusters=15,
                            affinity='cosine', linkage='average').fit(feat).labels_
                last_nmi.append(metrics.normalized_mutual_info_score(labels,V))
                last_ri.append(metrics.adjusted_rand_score(labels,V))
            elif pos>=7 and pos<=8:
                os.chdir(inp+'/'+f)
                print pos
                if pos == 7:
                    A1 = pickle.load(open('autoencoders_500.pkl','rb'))
                    A2 = pickle.load(open('autoencoders_2000.pkl','rb'))
                    A3 = pickle.load(open('autoencoders_10.pkl','rb'))
                else:
                    A1 = pickle.load(open('autoencoders_500.pkl','rb'))
                    A2 = pickle.load(open('autoencoders_2000.pkl','rb'))
                    A3 = pickle.load(open('autoencoders_100.pkl','rb'))
                feat = A3.get_output(A2.get_hidden(A1.get_hidden(X)))
                V = AgglomerativeClustering(n_clusters=15,
                            affinity='cosine', linkage='average').fit(feat).labels_
                last_nmi.append(metrics.normalized_mutual_info_score(labels,V))
                last_ri.append(metrics.adjusted_rand_score(labels,V))
        os.chdir(inp)
        print last_ri
        print last_nmi
        np.save('eval_nmi_dec_15.npy',last_nmi)
        np.save('eval_ri_dec_15.npy',last_ri)
