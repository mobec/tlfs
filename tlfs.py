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

from dataset.datasets import DataSet
from models.autoencoder import Autoencoder
from util.plot import Plotter
import json

import numpy as np
np.random.seed(4)


# from tensorflow.python import keras as k
#
#
# img = k.preprocessing.image.load_img("/home/mob/Desktop/elephant.jpg", target_size=(224, 224))
# x = k.preprocessing.image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = k.applications.vgg19.preprocess_input(x)
#
# ae = Autoencoder(input_shape=(224, 224, 3))
# ae.train(0)
# y = ae.model.predict(x)
#
# # y += np.array([103.939, 116.779, 123.68])
#
# y = np.squeeze(y, axis=0)
# img = k.preprocessing.image.array_to_img(y)
# k.preprocessing.image.save_img("/home/mob/Desktop/elephant_pred.jpg", img)

normalization_factor = 3.172111148032056 * 3.0 # Two sigma deviation
normalization_factor *= 2.0 / 255.0 #  caffe style scaling to R8G8B8 (-128, 127)
normalization_shift = np.array([0.02614982, 0.11674846,  0.        ]) # negative mean

dataset = DataSet()
dataset.load(path="/home/mob/Desktop/Dataset", blocks=["velocity"], shuffle=True, norm_factors={"velocity": normalization_factor}, norm_shifts={"velocity": normalization_shift})
dataset.train.velocity.print_data_properties()
ae = Autoencoder(input_shape=(128, 128, 3))
hist = ae.train(10, dataset=dataset, batch_size=8,  augment=True)
ae.save_model(path="/home/mob/Desktop")
if hist:
    with open("/home/mob/Desktop/hist.json", 'w') as f:
        json.dump(hist.history, f)
# 11.6532

plot = Plotter()
orig = dataset.test.velocity
pred = ae.predict(orig, batch_size=8)
orig.denormalize()
orig.print_data_properties()
pred *= normalization_factor
pred -= normalization_factor
plot.plot_vector_field(orig[50], pred[50], title="Comp", scale=300.0)

plot.plot_vector_field(orig[150], pred[150], title="Comp", scale=300.0)

plot.plot_vector_field(orig[250], pred[250], title="Comp", scale=300.0)

if hist:
    plot.plot_history(hist.history, log_y=True)

plot.show(True)
