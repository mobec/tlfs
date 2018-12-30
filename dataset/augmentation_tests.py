###############################################################################
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
###############################################################################

import unittest

import numpy as np
from . import augmentation as aug


class TestAugmentation(unittest.TestCase):
    # def test_rotate(self):
    #     # Three dimensions
    #     x = np.random.rand(1, 2, 2, 1)
    #     y = aug.rotate(x, quat=np.array([-0.7071068, 0, 0, 0.7071068]))
    #     self.assertFalse(np.allclose(x, y))
    #     y = aug.rotate(y, quat=np.array([0.7071068, 0, 0, 0.7071068]))
    #     print(x)
    #     print(y)
    #     self.assertTrue(np.allclose(x, y))
    #     # Two dimensions

    def test_rotate90(self):
        # Three dimensions
        x = np.random.rand(4, 32, 32, 32, 3)
        y = aug.rotate90(x, axes=(0, 1), k=1)
        self.assertFalse(np.allclose(x, y))
        y = aug.rotate90(y, axes=(1, 0), k=1)
        self.assertTrue(np.allclose(x, y))

        # Two dimensions
        x = np.random.rand(4, 32, 32, 3)
        y = aug.rotate90(x, axes=(0, 1), k=1)
        self.assertFalse(np.allclose(x, y))
        y = aug.rotate90(y, axes=(1, 0), k=1)
        self.assertTrue(np.allclose(x, y))

    def test_flip(self):
        # Three dimensions
        x = np.random.rand(4, 32, 32, 32, 3)
        y = aug.flip(x, axes=(0, 1, 2))
        self.assertFalse(np.allclose(x, y))
        y = aug.flip(y, axes=(0, 1, 2))
        self.assertTrue(np.allclose(x, y))

        # Two dimensions
        x = np.random.rand(1, 32, 32, 3)
        y = aug.rotate90(x, axes=(0, 1))
        self.assertFalse(np.allclose(x, y))
        y = aug.rotate90(y, axes=(1, 0))
        self.assertTrue(np.allclose(x, y))

    def test_scale(self):
        # Three dimensions
        x = np.random.rand(4, 32, 32, 32, 3)
        aug.scale(x, 2)
        # Two dimensions
        x = np.random.rand(1, 32, 32, 3)
        aug.scale(x, 2)

    def test_random_tile(self):
        x = np.random.rand(3, 32, 32, 3)
        y = aug.random_tile(x, shape=(8, 8), seed=1337)
        self.assertEqual(y.shape, (3, 8, 8, 3))
        y_check = aug.random_tile(x, shape=(8,8), seed=1337)
        self.assertTrue(np.allclose(y, y_check))

if __name__ == '__main__':
    unittest.main()
