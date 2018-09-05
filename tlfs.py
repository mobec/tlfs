from dataset.datasets import DataSet
from models.autoencoder import Autoencoder
from util.plot import Plotter
import json

import numpy as np
np.random.seed(4)

normalization_factor = 3.172111148032056 * 3.0 # Two sigma deviation
normalization_factor *= 2.0 / 255.0 #  caffe style scaling to R8G8B8 (-128, 127)
normalization_shift = np.array([0.02614982, 0.11674846,  0.        ]) # negative mean

dataset = DataSet()
dataset.load(path="/home/mob/Desktop/Dataset", blocks=["velocity"], shuffle=True, norm_factors={"velocity": normalization_factor}, norm_shifts={"velocity": normalization_shift})
dataset.train.velocity.print_data_properties()
ae = Autoencoder(input_shape=(128, 128, 3))
hist = ae.train(10, dataset=dataset, batch_size=8,  augment=False)
ae.save_model(path="/home/mob/Desktop")
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

plot.plot_history(hist.history, log_y=True)

plot.show(True)
