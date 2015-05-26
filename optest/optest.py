import theano
from theano import scalar
from theano.tensor import elemwise




class upgrade_to_float(object):
    def __new__(self, *types):
        """
        Upgrade any int types to float32 or float64 to avoid losing precision.
        """
        conv = {scalar.int8: scalar.float32,
                scalar.int16: scalar.float32,
                scalar.int32: scalar.float64,
                scalar.int64: scalar.float64,
                scalar.uint8: scalar.float32,
                scalar.uint16: scalar.float32,
                scalar.uint32: scalar.float64,
                scalar.uint64: scalar.float64}
        return [scalar.get_scalar_type(scalar.Scalar.upcast(conv.get(t, t))) for t in types]


class ScalarTransferFunction(scalar.ScalarOp):
    nin = 2
    nout = 2

    def impl(self, x_re, x_im):
        return (1.0 / x_re + 1.0, x_im)

    def c_code(self, node, name, inp, out, sub):
        x_re, x_im = inp
        z_re, z_im = out

        if node.inputs[0].type == scalar.float32:
            return """%(z_re)s = 1.0f / %(x_re)s + 1.0f;
                      %(z_im)s = %(x_im)s;""" % locals()
        elif node.inputs[0].type == scalar.float64:
            return """%(z_re)s = 1.0 / %(x_re)s + 1.0;
                      %(z_im)s = %(x_im)s;""" % locals()
        else:
            raise NotImplementedError('only floatingpoint is implemented')

    def c_code_cache_version(self):
        v = super(ScalarTransferFunction, self).c_code_cache_version()
        if v:
            return (3,) + v
        else:
            return v


scalar_transferfunction = ScalarTransferFunction(upgrade_to_float, name='scalar_transferfunction')
transferfunction = elemwise.Elemwise(scalar_transferfunction, name='transferfunction')



class ScalarTransferFunctionOne(scalar.ScalarOp):
    nin = 1
    nout = 1

    def impl(self, x_re):
        return (1.0 / x_re + 1.0,)

    def c_code(self, node, name, inp, out, sub):
        x_re, = inp
        z_re, = out

        if node.inputs[0].type == scalar.float32:
            return """%(z_re)s = 1.0f / %(x_re)s + 1.0f;""" % locals()
        elif node.inputs[0].type == scalar.float64:
            return """%(z_re)s = 1.0 / %(x_re)s + 1.0;""" % locals()
        else:
            raise NotImplementedError('only floatingpoint is implemented')

    def c_code_cache_version(self):
        v = super(ScalarTransferFunctionOne, self).c_code_cache_version()
        if v:
            return (3,) + v
        else:
            return v


scalar_transferfunctionone = ScalarTransferFunctionOne(upgrade_to_float, name='scalar_transferfunctionone')
transferfunctionone = elemwise.Elemwise(scalar_transferfunctionone, name='transferfunctionone')


class ScalarTransferFunctionTwo(scalar.ScalarOp):
    nin = 2
    nout = 1

    def impl(self, x, y):
        return (x+y)

    def c_code(self, node, name, inp, out, sub):
        x,y = inp
        z_re, = out

        if node.inputs[0].type == scalar.float32 or node.inputs[0].type == scalar.float64:
            return """%(z_re)s = %(x)s + %(y)s;""" % locals()
        else:
            raise NotImplementedError('only floatingpoint is implemented')

    def c_code_cache_version(self):
        v = super(ScalarTransferFunctionTwo, self).c_code_cache_version()
        if v:
            return (5,) + v
        else:
            return v


scalar_transferfunctiontwo = ScalarTransferFunctionTwo(scalar.upcast_out, name='scalar_transferfunctiontwo')
transferfunctiontwo = elemwise.Elemwise(scalar_transferfunctiontwo, name='transferfunctionotwo')



