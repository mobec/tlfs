#
#   Copyright 2018 Moritz Becher
#
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import keras
import keras.regularizers
import keras.backend as K
import tensorflow as tf

class ConvolutionOrthogonality(keras.regularizers.Regularizer):
    def __init__(self, factor: float = 1.0):
        self.factor =  K.cast_to_floatx(factor)

    def __call__(self, kernel):
        ks = tf.split(kernel, kernel.shape[-1], axis=-1)
        result = 0.0
        for k in ks:
            k = K.squeeze(k, -1)
            # 1.) Flatten the weight matrix
            w = K.reshape(k, shape=(-1,))
            # 2.) construct a square matrix consisting of w repeated in the rows and clear the diagonal
            o = K.squeeze(K.repeat(K.expand_dims(w, -1) ,w.shape[0]), -1) - K.eye(w.shape[0].value) * w
            # 3.) the regularizer is the L1 norm of the product divided by two, to account for double entries from the
            # symmetric matrix o
            w = K.expand_dims(w, -1)
            result += self.factor / 2.0 * K.sum(K.abs(K.dot(o, w)))
        return result

    def get_config(self):
        return {'factor': float(self.factor)}


def ortho(factor: float = 1.0):
    return ConvolutionOrthogonality(factor)
