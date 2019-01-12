###############################################################################
#
#   Copyright 2018 Moritz Becher
#
#   abstract network layout class
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
###############################################################################


import tensorflow as tf
import keras as k
import extensions as e
import numpy as np

from models.architecture import Network

def cast_shape_to(shape, type):
    out_shape= []
    for i in shape:
        try:
            out_shape.append(type(i))
        except:
            out_shape.append(None)
    return out_shape

class VGG19(Network):
    def _init_vars(self, **kwargs):
        self.init_func = "glorot_normal"
        self.adam_epsilon = None #1e-8 # 1e-3
        self.adam_learning_rate = 0.00001 # higher values tend to overshoot in the beginning
        self.adam_weight_decay = 0.005#1e-5
        self.input_shape = kwargs.get("input_shape", (64, 64, 64, 1))
        self.classes = kwargs.get("classes", 1000)
        self.loss = k.losses.categorical_crossentropy
        self.metrics = [k.metrics.categorical_accuracy]
        self.variational_ae = False
        self.kl_beta = 1e-5
        self.tensorflow_seed = kwargs.get("tensorflow_seed", 1337)
        self.model = None
        self.ortho_regularizer_strength = kwargs.get("ortho_strength", 0.01)
        self.kernel_regularizer = None
        if kwargs.get("ortho_regularizer", False):
            self.kernel_regularizer = e.regularizers.ortho(self.ortho_regularizer_strength)
        tf.set_random_seed(self.tensorflow_seed)
        np.random.seed(1337)

    def _init_optimizer(self, epochs=1):
        self.optimizer = k.optimizers.Adam(lr=self.adam_learning_rate, epsilon=self.adam_epsilon, decay=self.adam_weight_decay)
        return self.optimizer

    def _build_model(self):
        self.add_custom_object(e.regularizers.ConvolutionOrthogonality)

        x = k.layers.Input(shape=self.input_shape)

        # Block 1
        h = k.layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block1_conv1')(x)
        h = k.layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block1_conv2')(h)
        h = k.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(h)

        # Block 2
        h = k.layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv1')(h)
        h = k.layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv2')(h)
        h = k.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(h)

        # Block 3
        h = k.layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv1')(h)
        h = k.layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv2')(h)
        h = k.layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv3')(h)
        h = k.layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv4')(h)
        h = k.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(h)

        # Block 4
        h = k.layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv1')(h)
        h = k.layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv2')(h)
        h = k.layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv3')(h)
        h = k.layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv4')(h)
        h = k.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(h)

        # Block 5
        h = k.layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv1')(h)
        h = k.layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv2')(h)
        h = k.layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv3')(h)
        h = k.layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv4')(h)
        h = k.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(h)

        # Classification block
        h = k.layers.Flatten(name='flatten')(h)
        h = k.layers.Dense(4096, activation='relu', name='fc1')(h)
        h = k.layers.Dense(4096, activation='relu', name='fc2')(h)
        y = k.layers.Dense(self.classes, activation='softmax', name='predictions')(h)

        self.model = k.models.Model(inputs=x, outputs=y)

    def _compile_model(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

    def save_model(self, path):
        # search for layers with custom regularizer and remove it from them
        layers_with_ortho = []
        for layer in self.model.layers:
            if hasattr(layer, "kernel_regularizer") and isinstance(layer.kernel_regularizer, e.regularizers.ConvolutionOrthogonality):
                factor = layer.kernel_regularizer.factor
                layers_with_ortho.append((layer, factor))
                layer.kernel_regularizer = None
        # save the model without the custom regularizer
        super().save_model(path)
        # restore the original regularizer
        for layer, factor in layers_with_ortho:
            layer.kernel_regularizer = e.regularizers.ConvolutionOrthogonality(factor)

    def _train(self, epochs, **kwargs):
        if epochs == 0:
            return None

        images, labels = kwargs["data"]
        batch_size = kwargs.get("batch_size", 32)
        augment = kwargs.get("augment", False)
        hist = None
        try:
            hist = self.model.fit(
                x=images,
                y=labels,
                validation_split=0.1,
                epochs=epochs
            )
        except KeyboardInterrupt:
            print("Interrupted by user")
        return hist

    def predict(self, x, batch_size):
        return self.model.predict(x, batch_size=batch_size)
