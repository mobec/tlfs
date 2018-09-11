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

import numpy as np
import scipy
import scipy.ndimage
from typing import Tuple
from collections import namedtuple


def __resolution(a: np.ndarray) -> Tuple:
    """
    Gives the resolution tuple. Effectively shape[1:-1]
    :param a: The input array
    :return: A tuple of the resolution shape
    """
    return a.shape[1:-1]


def __dim(a: np.ndarray) -> int:
    """
    Dimensionality of the given data set numpy array
    :param a: array like with shape (num_samples, +{res,} channels)
    :return: the number of {res,} axes
    """
    return len(__resolution(a))


def __is_vector_field(a: np.ndarray, channel_axis: int=-1) -> bool:
    """
    Returns true if the given numpy array is a vector field, such as velocity
    :param a: array like
    :param channel_axis: the axis of the channels
    :return: True if the channel axis is larger than 1
    """
    return a.shape[channel_axis] > 1


def __split_by_channels(a: np.ndarray, channel_axis: int=-1) -> list:
    """
    Splits a given numpy array along the channel axis.
    Eg: shape (1, 64, 64, 3) to [(1, 64, 64, 1), (1, 64, 64, 1), (1, 64, 64, 1)]
    :param a: array like
    :param channel_axis: optional
    :return:
    """
    return np.split(a, a.shape[channel_axis], channel_axis)


def rotate90(data: np.ndarray, axes: Tuple, k: int = 1) -> np.ndarray:
    """
    rotate the frame by 90 degrees from the first axis counterclockwise to the second
        axes: 2 int, from axis to axis; see np.rot90
            0,1,2 -> z,y,x
    :param data: an array of data
    :param axes: axes of the plane of rotation
    :param k: number of times the rotation is performed (can be negative)
    :return: rotated array
    """
    if len(axes) != 2:
        raise ValueError('need 2 axes for rotate90.')
    axes = [x + 1 for x in axes]

    data = np.rot90(data, k=k, axes=axes)

    if __is_vector_field(data):
        data = __rotate90_vectors(data, axes)

    return data


def __rotate90_vectors(data: np.ndarray, axes: Tuple, channel_layout: list=None):
    """
    Rotate the vectors in a numpy vector field
    :param data: numpy array
    :param axes: the axes defining the rotation plane
    :param channel_layout: optional swizzle
    :return: field containing rotated vectors
    """

    if len(axes) != 2:
        raise ValueError('need 2 axes for rotate90.')

    if not __is_vector_field(data):
        raise ValueError("Data is not a vector field")

    if not channel_layout:
        channel_layout = range(data.shape[-1])
    v = channel_layout

    channels = __split_by_channels(data)
    channels[v[-axes[0] + 2]], channels[v[-axes[1] + 2]] = -channels[v[-axes[1] + 2]], channels[v[-axes[0] + 2]]
    return np.concatenate(channels, -1)


def flip(data: np.ndarray, axes: Tuple=None) -> np.ndarray:
    """
    flip low and high data (single frame/tile) along the specified axes
        low, high: data format: (z,x,y,c)
        axes: list of axis indices 0,1,2-> z,y,x
    :param data: numpy array
    :param axes: data axes to be flipped. default: flip all
    :return: numpy array flipped along its axes
    """
    # axis: 0,1,2 -> z,y,x
    if not axes:
        axes = range(__dim(data))
    axes = [x + 1 for x in axes]

    # flip tiles/frames
    for axis in axes:
        data = np.flip(data, axis)

    if __is_vector_field(data):
        __flip_vectors(data, axes)

    return data


def __flip_vectors(data: np.ndarray, axes: Tuple, channel_layout: list=None) -> np.ndarray:
    """
    flip velocity vectors along the specified axes
        low: data with velocity to flip (4 channels: d,vx,vy,vz)
        axes: list of axis indices 0,1,2-> z,y,x
    :param data: numpy array
    :param axes: data axes to be flipped
    :param channel_layout: optional swizzle for channels
    :return: numpy array with flipped vectors
    """

    if not channel_layout:
        channel_layout = range(data.shape[-1])
    v = channel_layout

    # !axis order: data z,y,x
    channels = __split_by_channels(data)

    if 2 in axes:  # flip vel x
        channels[v[0]] *= (-1)
    if 1 in axes:
        channels[v[1]] *= (-1)
    if 0 in axes and len(v) == 3:
        channels[v[2]] *= (-1)

    return np.concatenate(channels, -1)


def scale(data: np.ndarray, factor: int) -> np.ndarray:
    """
    changes frame resolution to round((factor) * (original resolution))
    :param data: numpy array
    :param factor: scale factor >= 1
    :return: field with resolution (factor * original resolution)
    """
    # only same factor in every dim for now. how would it affect vel scaling?
    # check for 2D
    if __dim(data) == 2:
        scale = [1, factor, factor, 1]
    else:
        scale = [1, factor, factor, factor, 1]

    # changes the size of the frame. should work well with getRandomTile(), no bounds needed
    data = scipy.ndimage.zoom(data, scale, order=1, mode='constant', cval=0.0)

    # necessary?
    if __is_vector_field(data):
        data = __scale_vectors(data, factor)
    # data = self.special_aug(data, AOPS_KEY_SCALE, factor)

    return data


def __scale_vectors(data: np.ndarray, factor: int, channel_layout: list=None) -> np.ndarray:
    """
    Scale the vectors inside a numpy vector field
    :param data: The input array
    :param factor: The scaling factor
    :param channel_layout: An optional layout by which to swizzle the vectors
    :return: The array with scaled vector components
    """

    # scale vel? vel*=factor
    channels = __split_by_channels(data)

    if not channel_layout:
        channel_layout = range(data.shape[-1])
    v = channel_layout

    channels[v[0]] *= factor
    channels[v[1]] *= factor
    if len(v) == 3:
        channels[v[2]] *= factor

    return np.concatenate(channels, -1)


def random_tile(a: np.ndarray, shape: Tuple, seed: int) -> np.ndarray:
    """
    Select a random tile with specified shape out of the input.
    :param a: The input array
    :param shape: The shape of the tile
    :return: A numpy array of the specified tile shape
    """

    # generate random slicing windows
    dimensions = __dim(a)
    window_max_start = np.array(__resolution(a)) - np.array(shape)
    window_min_start = np.array([0] * dimensions)
    np.random.seed(seed)
    low = np.array([np.random.randint(min, max, a.shape[0]) for min, max in zip(window_min_start, window_max_start)]).T
    high = low + np.array(shape, dtype=int)

    # slice tiles
    tiles = np.split(a, a.shape[0], 0)
    for i, tile in enumerate(tiles):
        if dimensions == 2:
            tiles[i] = tile[:, low[i, 0]:high[i, 0], low[i, 1]:high[i, 1], :]
        elif dimensions == 3:
            tiles[i] = tile[:, low[i, 0]:high[i, 0], low[i, 1]:high[i, 1], low[i, 2]:high[i, 2], :]
    return np.concatenate(tiles, 0)


# def rotate(a: np.ndarray, quat: np.ndarray=np.random.normal(size=4)):
#     """
#     random uniform rotation of low and high data of a given frame
#     :param a: a numpy data array
#     :return: a random rotation of the field
#     """
#     # check if single frame
#
#     # 2D:
#     if __dim(a) == 2:
#         theta = np.pi * np.random.uniform(0, 2)
#         rotation_matrix = np.array([[1, 0, 0, 0],
#                                     [0, np.cos(theta), -np.sin(theta), 0],
#                                     [0, np.sin(theta), np.cos(theta), 0],
#                                     [0, 0, 0, 1]])
#
#     # 3D:
#     elif __dim(a) == 3:
#         # random uniform rotation in 3D
#         #quat = np.random.normal(size=4)
#         quat /= np.linalg.norm(quat)
#
#         q = np.outer(quat, quat) * 2
#         rotation_matrix = np.array([[1 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0],
#                                     [q[1, 2] + q[3, 0], 1 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0],
#                                     [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1 - q[1, 1] - q[2, 2], 0],
#                                     [0, 0, 0, 1]])
#
#     if __is_vector_field(a):
#         a = __rotate_vectors(a, rotation_matrix)
#     #data = self.special_aug(data, AOPS_KEY_ROTATE, rotation_matrix)
#
#     return __apply_transform(a, rotation_matrix.T)
#
# # def rotate_simple(low, high, angle):
# #     '''
# #         use a different method for rotation. about 30-40% faster than with rotation matrix, but only one axis.
# #     '''
# #     if len(low.shape) != 4 or len(high.shape) != 4:
# #         self.TCError('Data shape mismatch.')
# #     # test rot around z (axis order z,y,x,c)
# #     low = scipy.ndimage.rotate(low, angle, [1, 2], reshape=False, order=self.interpolation_order,
# #                                mode=self.fill_mode, cval=1.0)
# #     high = scipy.ndimage.rotate(high, angle, [1, 2], reshape=False, order=self.interpolation_order,
# #                                 mode=self.fill_mode, cval=1.0)
# #     return low, high
#
#
# def __rotate_vectors(data: np.ndarray, rotation_matrix: np.ndarray, channel_layout: list=None):
#     """
#     Rotate the vectors in a numpy vector field with the rotation of the whole field
#     :param data: vector field
#     :param rotation_matrix: the matrix of the rotation operation
#     :param channel_layout: an optional swizzle of the vectors
#     :return: the vector field with rotated vectors
#     """
#     if not __is_vector_field(data):
#         raise ValueError("Data is not a vector field")
#
#     if not channel_layout:
#         channel_layout = range(data.shape[-1]) # identity layout
#     v = channel_layout
#
#     channels = __split_by_channels(data)
#     if len(v) == 3:  # currently always ends here!! even for 2D, #z,y,x to match rotation matrix
#         vel = np.stack([channels[v[2]].flatten(), channels[v[1]].flatten(), channels[v[0]].flatten()])
#         vel = rotation_matrix[:3, :3].dot(vel)
#         channels[v[2]] = np.reshape(vel[0], channels[v[2]].shape)
#         channels[v[1]] = np.reshape(vel[1], channels[v[1]].shape)
#         channels[v[0]] = np.reshape(vel[2], channels[v[0]].shape)
#     if len(v) == 2:
#         vel = np.concatenate([channels[v[1]], channels[v[0]]], -1)  # y,x to match rotation matrix
#         shape = vel.shape
#         vel = np.reshape(vel, (-1, 2))
#         vel = np.reshape(rotation_matrix[1:3, 1:3].dot(vel.T).T, shape)
#         vel = np.split(vel, 2, -1)
#         channels[v[1]] = vel[0]
#         channels[v[0]] = vel[1]
#
#     return np.concatenate(channels, -1)
#
#
# def __apply_transform(data, transform_matrix):
#     """
#     Apply an affine transformation matrix to the data
#     :param data: The input array
#     :param transform_matrix: The affine transformation matrix
#     :return: The transformed array
#     """
#     data_dim = __dim(data)
#
#     data = np.split(data, data.shape[0], 0)
#     for i, d in enumerate(data):
#         # match shape to what is expected by transform
#         d = np.squeeze(d, axis=0)
#         if data_dim == 2:
#             # expand the two dim field to be three dimensional
#             d = np.expand_dims(d, axis=2)
#         channels = __split_by_channels(d)
#         for i, c in enumerate(channels):
#             channels[i] = np.squeeze(c, axis=-1)
#
#
#         # set transform to center; from fluiddatagenerator.py
#         # offset = np.array(d.shape) / 2 - np.array([0.5, 0.5, 0.5, 0])
#         # offset_matrix = np.array([[1, 0, 0, offset[0]], [0, 1, 0, offset[1]], [0, 0, 1, offset[2]], [0, 0, 0, 1]])
#         # reset_matrix = np.array([[1, 0, 0, -offset[0]], [0, 1, 0, -offset[1]], [0, 0, 1, -offset[2]], [0, 0, 0, 1]])
#         # transform_matrix = np.dot(np.dot(offset_matrix, transform_matrix), reset_matrix)
#
#         channel_data = [scipy.ndimage.interpolation.affine_transform(
#             channel,
#             matrix=transform_matrix,
#             order=1,
#             mode='constant',
#             cval=0.) for channel in channels]
#
#         # bring data back to original shape
#         for i, c in enumerate(channel_data):
#             channel_data[i] = np.expand_dims(c, axis=3)
#         d = np.concatenate(channel_data, axis=3)
#         if data_dim == 2:
#             # remove added dimension
#             d = np.squeeze(d, axis=2)
#         d = np.expand_dims(d, axis=0)
#
#         data[i] = d
#
#     return np.concatenate(data, 0)
