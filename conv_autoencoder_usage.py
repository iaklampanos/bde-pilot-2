data_dict = netCDF_subset('/mnt/disk1/thanasis/data/train_modified.nc',
                          [700], ['UU'], lvlname='num_metgrid_levels', timename='Times')
items = [data_dict.extract_data()]
items = np.array(items)
print items.shape
ds = Dataset_transformations(items, 1000, items.shape)
ds.twod_transformation()
ds.normalize()
# ds.shift()
print ds.get_items().shape
X_train = np.transpose(ds.get_items())
print '--------------'
print X_train.shape
X_train = X_train.reshape((11688, 1, 64, 64))
X_train = X_train.astype(np.float32)
print X_train.shape
# we need our target to be 1 dimensional
X_out = X_train.reshape((X_train.shape[0], -1))
print X_out.shape
ae.fit(X_train, X_out)
print


import pickle
import sys
sys.setrecursionlimit(10000)

pickle.dump(ae, open('mconv_ae.pkl', 'w'))

X_train_pred = ae.predict(X_train).reshape(11688, 1, 64, 64)

np.save('X_train.npy', X_train)
np.save('X_train_pred.npy', X_train_pred)

encode_layer_index = map(lambda pair: pair[0], ae.layers).index('encode_layer')
encode_layer = ae.get_all_layers()[encode_layer_index]


X_encoded = encode_input(X_train)
print X_encoded.shape
exit(-1)
