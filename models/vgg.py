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

class VGG(Network):
    def _init_vars(self, **kwargs):
        self.init_func = "glorot_normal"
        self.adam_epsilon = None #1e-8 # 1e-3
        self.adam_learning_rate = 0.0001 # higher values tend to overshoot in the beginning
        self.adam_weight_decay = 0.000#1e-5
        self.input_shape = kwargs.get("input_shape", (64, 64, 64, 1))
        self.loss = "mse"
        self.metrics = ["mae"]
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

        # ----------------------------------------------------------------------------------
        x = k.layers.Input(shape=self.input_shape)
        # ----------------------------------------------------------------------------------
        h = k.layers.Conv2D(64,  (3, 3), padding='same', trainable=True, kernel_regularizer=self.kernel_regularizer)(x)
        h = k.layers.Activation('relu')(h)
        # ----------------------------------------------------------------------------------
        h = k.layers.Conv2D(64, (3, 3), padding='same', trainable=True, kernel_regularizer=self.kernel_regularizer)(h)
        h = k.layers.Activation('relu')(h)
        # ----------------------------------------------------------------------------------
        h = k.layers.MaxPool2D((2,2))(h)

        #----------------------------------------------------------------------------------
        h = k.layers.Conv2D(128, (3, 3), padding='same', trainable=True, kernel_regularizer=self.kernel_regularizer)(h)
        h = k.layers.Activation('relu')(h)
        # ----------------------------------------------------------------------------------
        h = k.layers.Conv2D(128, (3, 3), padding='same', trainable=True, kernel_regularizer=self.kernel_regularizer)(h)
        h = k.layers.Activation('relu')(h)
        # ----------------------------------------------------------------------------------
        h = k.layers.MaxPool2D((2, 2))(h)

        #----------------------------------------------------------------------------------
        h = k.layers.Conv2D(256, (3, 3), padding='same', trainable=True)(h)
        h = k.layers.Activation('relu')(h)
        # # ----------------------------------------------------------------------------------
        # h = k.layers.Conv2D(256, (3, 3), padding='same', trainable=True)(h)
        # h = k.layers.Activation('relu')(h)
        # # ----------------------------------------------------------------------------------
        # h = k.layers.Conv2D(256, (3, 3), padding='same', trainable=True)(h)
        # h = k.layers.Activation('relu')(h)
        # # # ----------------------------------------------------------------------------------
        # # h = k.layers.Conv2D(256, (3, 3), padding='same')(h)
        # # h = k.layers.Activation('relu')(h)
        # #
        # #
        # # # ----------------------------------------------------------------------------------
        # # h = k.layers.Conv2DTranspose(256, (3, 3), padding='same')(h)
        # # h = k.layers.Activation('relu')(h)
        # # ----------------------------------------------------------------------------------
        # h = k.layers.Conv2DTranspose(256, (3, 3), padding='same')(h)
        # h = k.layers.Activation('relu')(h)
        # # ----------------------------------------------------------------------------------
        # h = k.layers.Conv2DTranspose(256, (3, 3), padding='same')(h)
        # h = k.layers.Activation('relu')(h)
        # ----------------------------------------------------------------------------------
        h = k.layers.Conv2DTranspose(128, (3, 3), padding='same')(h)
        h = k.layers.Activation('relu')(h)

        #----------------------------------------------------------------------------------
        #h = e.layers.InvMaxPool2D((2, 2))(h)
        h = k.layers.UpSampling2D((2, 2))(h)
        # ----------------------------------------------------------------------------------
        h = k.layers.Conv2DTranspose(128, (3, 3), padding='same', kernel_regularizer=self.kernel_regularizer)(h)
        h = k.layers.Activation('relu')(h)
        # ----------------------------------------------------------------------------------
        h = k.layers.Conv2DTranspose(64, (3, 3), padding='same', kernel_regularizer=self.kernel_regularizer)(h)
        h = k.layers.Activation('relu')(h)

        #----------------------------------------------------------------------------------
        #h = e.layers.InvMaxPool2D((2, 2))(h)
        h = k.layers.UpSampling2D((2, 2))(h)
        # ----------------------------------------------------------------------------------
        h = k.layers.Conv2DTranspose(64, (3, 3), padding='same', kernel_regularizer=self.kernel_regularizer)(h)
        h = k.layers.Activation('relu')(h)
        # ----------------------------------------------------------------------------------
        y = k.layers.Conv2DTranspose(3, (3, 3), padding='same', kernel_regularizer=self.kernel_regularizer)(h)
        y = k.layers.Conv2DTranspose(3, (3, 3), padding='same', kernel_regularizer=self.kernel_regularizer)(h)

        self.model = k.models.Model(inputs=x, outputs=y)

        vgg19 = k.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=self.input_shape, pooling=None, classes=1000)
        vgg19_weights = vgg19.get_weights()

        for i in range(len(vgg19_weights) // 2):
            weights = vgg19_weights[i*2]
            biases = vgg19_weights[i*2 + 1]
            print("Weight sum: {}".format(np.sum(weights)))
            print("Bias sum: {}".format(np.sum(biases)))

        vgg_layer_scales = [
            1.000021775577029,
            0.7820745278554599,
            4.506298490266108,
            1.520877112218829,
            1.7411676435890928,
            0.7973368912842906,
            1.1258171358619091,
            1.5871279482567102,
            2.211062450971975,
            0.8242409135760195,
            0.7673000726826673,
            0.5574201626639431,
            0.5702032871407169,
            0.4819161555954472,
            0.4929815244100853
        ]

        model_layer_idx = 0
        vgg19_layer_idx = 0
        # encoder weights
        for i in range(6):
            # search for the next conv2d layer
            while not self.model.layers[model_layer_idx].get_weights():
                model_layer_idx += 1
            # set weights from vgg
            weights = vgg19_weights[vgg19_layer_idx]
            print(weights.shape)
            # weights = weights / vgg_layer_scales[i]
            biases = vgg19_weights[vgg19_layer_idx + 1]
            #biases = biases / np.sum(biases)
            conv2d_weights = [weights, biases]
            self.model.layers[model_layer_idx].set_weights(conv2d_weights)
            vgg19_layer_idx += 2
            model_layer_idx += 1

        # decoder weights
        model_layer_idx = len(self.model.layers) - 1
        vgg19_layer_idx = 0
        for i in range(6):
            # search for the next conv2d transposed layer
            while not self.model.layers[model_layer_idx].get_weights():
                model_layer_idx -= 1
            # set weights from vgg
            weights = vgg19_weights[vgg19_layer_idx]
            # weights = weights / vgg_layer_scales[i]
            biases = vgg19_weights[vgg19_layer_idx + 1][:self.model.layers[model_layer_idx].get_weights()[1].shape[0]]
            #biases = biases / np.sum(biases)
            conv2d_T_weights = [weights, biases]
            self.model.layers[model_layer_idx].set_weights(conv2d_T_weights)
            vgg19_layer_idx += 2
            model_layer_idx -= 1

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

        dataset = kwargs["dataset"]
        batch_size = kwargs.get("batch_size", 32)
        augment = kwargs.get("augment", False)
        hist = None
        try:
            train_generator = dataset.train.generator(batch_size=batch_size, augment=augment, noise=False)
            train_steps_per_epoch = dataset.train.steps_per_epoch(batch_size=batch_size, augment=augment)
            val_generator = dataset.val.generator(batch_size=batch_size, augment=augment, noise=False)
            val_steps = dataset.val.steps_per_epoch(batch_size=batch_size, augment=augment)
            print(val_steps)
            hist = self.model.fit_generator(
                generator=train_generator,
                steps_per_epoch=train_steps_per_epoch,
                validation_data=val_generator,
                validation_steps=val_steps,
                epochs=epochs,
                max_queue_size=100
            )
        except KeyboardInterrupt:
            print("Interrupted by user")
        return hist

    def predict(self, x, batch_size):
        return self.model.predict(x, batch_size=batch_size)

