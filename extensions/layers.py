import tensorflow as tf
from tensorflow.python import keras as k

class InvMaxPool2D(k.layers.Layer):
    """ Layer for upsampling """

    def __init__(self, size=(2,2), mode = 'zeros'):
        """
        Imitate reverse operation of Max-Pooling by either placing original max values
        into a fixed postion of upsampled cell:
        [0.9] =>[[.9, 0],   (stride=2)
                [ 0, 0]]
        or copying the value into each cell:
        [0.9] =>[[.9, .9],  (stride=2)
                [ .9, .9]]
        :param net: 4D input tensor with [batch_size, width, heights, channels] axis
        :param stride:
        :param mode: string 'zeros' or 'copy' indicating which value to use for undefined cells
        :return:  4D tensor of size [batch_size, width*stride, heights*stride, channels]
        """
        self.size = tuple(size)
        # the mode with which the inverse max pooling will be performed
        assert mode in ['copy', 'zeros'], ("Mode must be either 'zeros' or 'copy'")
        self.mode = mode
        super(InvMaxPool2D, self).__init__()

    def get_output_shape_for(self, input_shape):
        width = self.size[0] * input_shape[2] if input_shape[2] is not None else None
        height = self.size[1] * input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0], width, height, input_shape[3])

    def call(self, x, mask=None):
        def _upsample_along_axis(x, axis, stride, mode):
            # get the dynamic shape
            batch_size = k.backend.shape(x)[0]
            shape = x.shape
            input_shape = [batch_size, int(shape[1]), int(shape[2]), int(shape[3])]
            # construct the output shape
            output_shape = list(input_shape)
            output_shape[axis] = output_shape[axis] * stride
            # create a tensor filled with zeros
            zeros = tf.zeros(input_shape, dtype=x.dtype) if mode == 'zeros' else x
            # add zeros 'behind' the input
            parts = [x] + [zeros for _ in range(stride - 1)]
            # insert stride zeros after input value
            merged = tf.concat(parts, min(axis + 1, 3))
            # reshape to desired output shape
            output = tf.reshape(merged, output_shape)
            return output
        x = _upsample_along_axis(x=x, axis=1, stride=self.size[0], mode=self.mode)
        x = _upsample_along_axis(x=x, axis=2, stride=self.size[1], mode=self.mode)
        return x

