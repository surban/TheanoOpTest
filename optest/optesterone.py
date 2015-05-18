import theano
import theano.tensor as T
import numpy as np

from optest import transferfunctionone

x_re = T.vector('x_re', theano.config.floatX)

z_re = transferfunctionone(x_re)

z_func = theano.function([x_re], [z_re], allow_input_downcast=True)

X = np.asarray([1 + 0j, 3+1j])
Z_re = z_func(np.real(X))
Z = Z_re

print "X=",X
print "Z=",Z




theano.printing.pydotprint(z_func, outfile="optesterone.png", var_with_name_simple=True)


