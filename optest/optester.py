import theano
import theano.tensor as T
import theano.printing
import numpy as np

from optest import transferfunction

x_re = T.vector('x_re', theano.config.floatX)
x_im = T.vector('x_im', theano.config.floatX)

z_re, z_im = transferfunction(x_re, x_im)

z_func = theano.function([x_re, x_im], [z_re, z_im], allow_input_downcast=True, mode='DebugMode')

X = np.asarray([1 + 0j, 3+1j])
Z_re, Z_im = z_func(np.real(X), np.imag(X))
Z = Z_re + Z_im * 1j

print "X=",X
print "Z=",Z

theano.printing.debugprint(Z_re)
theano.printing.pydotprint(z_func, outfile="optester.png", var_with_name_simple=True)

