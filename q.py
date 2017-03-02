import dataset_utils as utils
import numpy as np

train = np.load('X_train.npy')
pred = np.load('X_train_pred.npy')
train = train.reshape(11688,4096)
pred = pred.reshape(11688,4096)
for i in range(0,40):
    utils.plot_pixel_image(train[i],pred[i],64,64)
