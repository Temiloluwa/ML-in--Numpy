#import tensorflow as tf
#from tensorflow.keras.layers import SimpleRNN
from rnn import VanillaRNN
import numpy as np

embed_dims = 5
ts = 3
hs_dim = 4
out_dim = 5
X = np.arange(30).reshape((2, ts, embed_dims))
model = VanillaRNN(hs_dim, out_dim, "ones_initializer")
output = model(X)
print("outputshape", output[0].shape)
