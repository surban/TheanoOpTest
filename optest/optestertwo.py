import theano
import theano.tensor as T
import numpy as np

from optest import transferfunctiontwo

x_re = T.vector('x_re', theano.config.floatX)
y_re = T.vector('y_re', theano.config.floatX)
z_re = transferfunctiontwo(x_re,y_re)

z_func = theano.function([x_re, y_re], [z_re], allow_input_downcast=True)

X = np.asarray([1 + 0j, 3+1j])
Z_re = z_func(np.real(X), np.imag(X))
Z = Z_re

print "X=",X
print "Z=",Z

theano.printing.pydotprint(z_func, outfile="optestertwo.png", var_with_name_simple=True)
print z_func.maker.fgraph.toposort()


