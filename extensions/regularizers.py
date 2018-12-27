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
        self.factor = factor

    def __call__(self, kernel):
        ks = tf.split(kernel, kernel.shape[-1], axis=-1)
        result = K.variable(0.0, dtype="float32", name="result_accumulator")
        for k in ks:
            #k = K.squeeze(k, -1)
            # 1.) Flatten the weight matrix
            w = K.flatten(k)
            print(w.shape)
            # 2.) construct a square matrix consisting of w repeated in the columns and clear the diagonal
            o = K.squeeze(K.repeat(K.expand_dims(w, -1), w.shape[0]), -1) - K.eye(w.shape[0].value) * w
            # 3.) the regularizer is the L1 norm of the product divided by two, to account for double entries from the
            # symmetric matrix o
            result += K.cast_to_floatx(self.factor) / 2.0 * (K.sum(K.abs(tf.matmul(o, K.expand_dims(w, -1), transpose_a=True))))
            #+ K.abs(tf.norm(w, ord=2) - 1.0)
        return result

    def get_config(self):
        return {'factor': float(self.factor)}


def ortho(factor: float = 1.0):
    return ConvolutionOrthogonality(factor)
