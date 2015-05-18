import theano
import theano.tensor as T
import numpy as np

x = T.vector('x', theano.config.floatX)

z = T.nnet.sigmoid(x)

z_func = theano.function([x], [z], allow_input_downcast=True)

X = np.asarray([1, 3])
Z = z_func(X)

print "X=",X
print "Z=",Z



