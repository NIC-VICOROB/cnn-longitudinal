from lasagne.init import Constant
from lasagne.layers import MergeLayer


class WeightedSumLayer(MergeLayer):
    """
    Weighted sum layer
    The layer applies a weighted sum between all the inputs. The weights are trainable.
    Parameters
    ----------
    incomings : a list of :class:`Layer` instances
        The layers feeding into this layer.
    """
    def __init__(self, incomings, **kwargs):
        super(WeightedSumLayer, self).__init__(incomings, **kwargs)
        self.coeff_left = self.add_param(Constant(-1), (1,), name='coeff_left')
        self.coeff_right = self.add_param(Constant(1), (1,), name='coeff_right')

    def get_params(self, unwrap_shared=True, **tags):
        return [self.coeff_left, self.coeff_right]

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        left, right = inputs
        return left * self.coeff_left + right * self.coeff_right