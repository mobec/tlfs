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
import os
import pickle

from dataset.datasets import DataSet
from models.vgg import VGG
from util.plot import Plotter
import numpy as np
np.random.seed(4)

# normalization_factor = np.array([3.94457719, 4.94287727, 1.]) * 3.0  # 3.172111148032056 * 2.0 # Two sigma deviation
normalization_factor = np.array([4.28670895, 4.10170295, 1.]) * 44 # liquid
normalization_factor *= 1.0 / 255.0  # caffe style scaling to R8G8B8 (-128, 127)
normalization_shift = np.array([0.02614982, 0.11674846,  0.0]) # negative mean


def train_tlfs(dataset_path, model_path, epochs):
    dataset = DataSet()
    dataset.load(path=dataset_path, blocks=["velocity"], shuffle=False, norm_factors={"velocity": normalization_factor}, norm_shifts={"velocity": normalization_shift})

    hist = {}

    ae = VGG(input_shape=(None, None, 3))
    # if os.path.isfile(model_path):
    #     ae.load_model(path=model_path)

    hist = ae.train(epochs, dataset=dataset, batch_size=32,  augment=False)
    ae.save_model(path=model_path)

    return hist


def predict_test_data(dataset_path, model_path):
    dataset = DataSet()
    dataset.load(path=dataset_path, blocks=["velocity"], shuffle=False, norm_factors={"velocity": normalization_factor}, norm_shifts={"velocity": normalization_shift})

    ae = VGG(input_shape=(None, None, 3))

    ae.load_model(path=model_path)

    orig = dataset.test.velocity
    pred = ae.predict(orig.data, batch_size=8)
    orig.denormalize()
    pred *= normalization_factor
    pred -= normalization_factor

    return orig.data, pred


if __name__ == '__main__':

    try:
        import argparse
        import os
        import json

        parser = argparse.ArgumentParser(description="Train the tlfs model")
        parser.add_argument("-o", "--output", type=str, required=True, help="The output path")
        parser.add_argument("-d", "--dataset", type=str, required=True, help="The dataset path")
        parser.add_argument("--train", action="store_true", help="Train the model")
        parser.add_argument("--test", action="store_true", help="Test the model")
        parser.add_argument("--gui", action="store_true", help="Test the model")
        parser.add_argument("--epochs", type=int, default=50, help="The number of training epochs")
        parser.add_argument("model", type=str, help="The path to the model file (.h5)")
        args = parser.parse_args()

        os.makedirs(args.output, exist_ok=True)

        plot = Plotter()

        if args.train:
            hist = train_tlfs(args.dataset, args.model, args.epochs)
            if hist:
                with open(args.output + "/hist.json", 'w') as f:
                    json.dump(hist.history, f)
                plot.plot_history(hist.history, log_y=True)

        if args.test:
            originals, predictions = predict_test_data(args.dataset, args.model)
            for o, p in zip(originals, predictions):
                plot.plot_vector_field(o, p, title="Test Prediction", scale=300.0)

        if args.gui:
            plot.show(True)

        figures_path = args.output + "/figures"
        os.makedirs(figures_path, exist_ok=True)
        plot.save_figures(figures_path)
    finally:
        import util.pushover
        util.pushover.notify("Training complete")
