#******************************************************************************
#
#   Keras Gems
#   Copyright 2018 Moritz Becher, Lukas Prantl and Steffen Wiewel
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
#******************************************************************************

from abc import ABC, abstractmethod
import inspect
import os
import pickle

import numpy as np
import tensorflow as tf
from keras import backend as K
import keras as k


class Network(ABC):
    """ Superclass for all model architectures """
    #---------------------------------------------------------------------------------
    def __init__(self, **kwargs):
        # Initialize Variables (Hyperparams etc)
        self.model = None
        self._init_vars(**kwargs)
        # Optimizer
        self._init_optimizer(1)
        # # Build & Compile -> Call this by yourself when model is loaded by weights only
        # self._build_model()
        # self._compile_model()
        self.__custom_objects = {}

    def add_custom_object(self, class_type):
        self.__custom_objects[class_type.__name__] = class_type

    #---------------------------------------------------------------------------------
    def train(self, epochs, **kwargs):
        """ Trains and returns the training history """
        # Reset all random number generators to given seeds
        #requirements.reset_rng()
        np.random.seed(4213)
        tf.set_random_seed(3742)

        # Destroys the current TF graph and creates a new one.
        # Useful to avoid clutter from old models / extensions.
        if self.model is not None:
            del self.model
            self.model = None
        K.clear_session()

        # Recompile (in case of updated hyper parameters)
        self._init_optimizer(epochs)
        # TODO: remove this?
        self._build_model()
        self._compile_model()
        # Model Summary
        self.model.summary()
        self.print_attributes()
        # Train and return History
        if epochs != 0:
            history = self._train(epochs, **kwargs)
            return history
        return None

    #---------------------------------------------------------------------------------
    def save_model(self, path):
        # This is the hack to support custom objects in a loader that might not know about the objects definitions.
        # Unfortunately keras does not allow us to implement this in a more convenient way
        # custom_objects_path = os.path.dirname(path) + "/custom_objects.pickle"
        # with open(custom_objects_path, 'wb') as f:
        #     print(self.__custom_objects)
        #     pickle.dump(self.__custom_objects, f, pickle.HIGHEST_PROTOCOL)

        # currently custom regularizers are just not embedded into the saved model

        k.models.save_model(self.model, path)

    #---------------------------------------------------------------------------------
    def load_model(self, path, custom_objects=None):
        # if custom_objects is None:
        #     custom_objects = {}
        #
        # custom_objects_path = os.path.dirname(path) + "/custom_objects.pickle"
        # with open(custom_objects_path, 'rb') as f:
        #     pickled_custom_objects = pickle.load(f)
        #
        # pickled_custom_objects.update(custom_objects)

        self.model = k.models.load_model(path)

    #---------------------------------------------------------------------------------
    # Interface
    #---------------------------------------------------------------------------------
    @abstractmethod
    def _init_vars(self, **kwargs):
        """ Setup internal variables like hyper parameters """
        pass
    #---------------------------------------------------------------------------------
    @abstractmethod
    def _init_optimizer(self, epochs=1):
        """ Initializes and stores the optimizer """
        pass
    #---------------------------------------------------------------------------------
    @abstractmethod
    def _build_model(self):
        """ Build the network structure and store in self.model """
        pass
    #---------------------------------------------------------------------------------
    @abstractmethod
    def _compile_model(self):
        """ Calls compile on the model """
        pass
    #---------------------------------------------------------------------------------
    @abstractmethod
    def _train(self, epochs, **kwargs):
        """ Trains and returns the training history """
        return
    #---------------------------------------------------------------------------------
    @abstractmethod
    def predict(self, x, batch_size):
        """ Predict the results of X using the compiled model and return the result """
        return

    #---------------------------------------------------------------------------------
    # Functions
    #---------------------------------------------------------------------------------
    # e.g. params = {"learning_rate": 0.1, "use_bias": True, ...}
    def update_parameters(self, params):
        """ Update the member variables by passing in names and new values as a dict """
        for key, value in params.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                print("[WARNING] Key {} does not exist!".format(key))
    #---------------------------------------------------------------------------------
    def print_attributes(self):
        attribute_list = []
        max_length = 10
        for i in inspect.getmembers(self):
            if not i[0].startswith('_'):
                if not inspect.isclass(i[1]) and not inspect.ismethod(i[1]):
                    attribute_list.append(i)
                    max_length = max_length if len(i[0]) < max_length else len(i[0])
        for attr in attribute_list:
            print("{:{width}}: {}".format(attr[0], attr[1], width=max_length))
