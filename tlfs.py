from dataset.datasets import DataSet
from models.autoencoder import Autoencoder
from util.plot import Plotter
import json

import numpy as np
np.random.seed(4)

normalization_factor = 3.172111148032056 * 2.0 # Three sigma deviation

dataset = DataSet()
dataset.load(path="/home/mob/Desktop/Dataset", blocks=["velocity"], shuffle=True, norm_factors={"velocity": 2.0*normalization_factor}, norm_shifts={"velocity": normalization_factor})
ae = Autoencoder(input_shape=(128, 128, 3))
hist = ae.train(3, dataset=dataset, batch_size=8,  augment=False)
ae.save_model(path="/home/mob/Desktop")
with open("/home/mob/Desktop/hist.json", 'w') as f:
    json.dump(hist.history, f)
# 11.6532

plot = Plotter()
orig = dataset.test.velocity
pred = ae.predict(orig, batch_size=8)
orig.denormalize()
orig.print_data_properties()
pred *= 2.0 * normalization_factor
pred -= normalization_factor
plot.plot_vector_field(orig[0], pred[0], title="Comp", scale=normalization_factor * 20.0)

plot.plot_vector_field(orig[5], pred[5], title="Comp", scale=normalization_factor * 20.0)

plot.plot_vector_field(orig[10], pred[10], title="Comp", scale=normalization_factor * 20.0)

plot.show(True)
