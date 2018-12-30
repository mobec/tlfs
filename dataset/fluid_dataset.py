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

import numpy as np
from .datasets import DataSet


def load_fluid_dataset(dataset_path, fluid_type="smoke") -> DataSet:
    dataset = DataSet()
    dataset.load(path=dataset_path, blocks=["velocity"], shuffle=True, validation_split=0.1, test_split=0.1)
    normalization_shift = np.mean(dataset.train.velocity.data, axis=(0, 1, 2))
    normalization_factor = np.std(dataset.train.velocity.data, axis=(0, 1, 2))
    normalization_factor[-1] = 1.0
    normalization_factor *= 1.0 / 255.0
    dataset.train.velocity.normalize(shift=normalization_shift, factor=normalization_factor)
    dataset.val.velocity.normalize(shift=normalization_shift, factor=normalization_factor)
    dataset.test.velocity.normalize(shift=normalization_shift, factor=normalization_factor)
    return dataset
    # if fluid_type == "smoke":
    #     return load_smoke_dataset(dataset_path)
    # else:
    #     return load_liquid_dataset(dataset_path)


def load_liquid_dataset(dataset_path) -> DataSet:
    ## LIQUID
    #normalization_factor = np.array([88.57836914, 79.51068115, 1.0]) * 2
    normalization_factor = np.array([4.28670895, 4.10170295, 1.        ]) * 44

    normalization_factor *= 1.0 / 255.0  # caffe style scaling to R8G8B8 (-128, 127)
    normalization_shift = np.array([-0.01400219, 0.03284957,  0.        ])

    dataset = DataSet()
    dataset.load(path=dataset_path, blocks=["velocity"], shuffle=True, norm_factors={"velocity": normalization_factor},
                 norm_shifts={"velocity": normalization_shift}, validation_split=0.1, test_split=0.1)
    print("Liquid dataset min: {} max: {}".format(np.min(dataset.train.velocity.data, axis=(0,1,2)), np.max(dataset.train.velocity.data, axis=(0,1,2))))
    return dataset


def load_smoke_dataset(dataset_path) -> DataSet:
    # SMOKE
    #normalization_factor = np.array([36.60723367, 45.46171525,   1.0]) * 2  # min max normalization to [-0.5, 0.5]
    normalization_factor = np.array([3.83091661, 4.74718894,   1.0]) * 44  # std normalization to [-0.5, 0.5]

    normalization_factor *= 1.0 / 255.0  # caffe style scaling to R8G8B8 (-128, 127)
    normalization_shift = np.array([ -0.01714196, 0.08772996,  0.        ])  # negative mean

    dataset = DataSet()
    dataset.load(path=dataset_path, blocks=["velocity"], shuffle=True, norm_factors={"velocity": normalization_factor},
                 norm_shifts={"velocity": normalization_shift}, validation_split=0.1, test_split=0.1)
    print("Smoke dataset min: {} max: {}".format(np.min(dataset.train.velocity.data, axis=(0,1,2)), np.max(dataset.train.velocity.data, axis=(0,1,2))))
    return dataset