
# -*- coding: utf-8 -*-
"""
Created on Sunday June 12 16:33:21 2021

Author: Anjali Balagopal
Email: anjalibalagopal91@gmail.com

Code adapted from : https://www.programmersought.com/article/5408456601/ & https://github.com/wcfzl/3D-CNNs-for-Liver-Classification.git
"""

import tensorflow as tf

# first transform the matrix t into a one-dimensional matrix, and then change the form of the matrix
# tf_flatten pulls a into one dimension
def tf_flatten(arr):
    """Flatten tensor"""
    return tf.reshape(arr, [-1])


# Increase the dimension to 2 dimensions, then increase the dimension data, and then pull it into one dimension
def tf_repeat(arr, repeats):
    """TensorFlow version of np.repeat for 1D"""
    assert len(arr.get_shape()) == 1  # To ensure shape is 1D
    arr = tf.expand_dims(arr, -1)
    arr = tf.tile(arr, [1, repeats])  #Extend a (number of first dimension*1, number of second dimension*repeats)
    arr = tf_flatten(arr)
    return arr


def tf_repeat_2d(arr, repeats):
    """Tensorflow version of np.repeat for 2D"""
    assert len(arr.get_shape()) == 2
    a = tf.expand_dims(arr, 0)
    a = tf.tile(a, [repeats, 1, 1])
    return a


'''
tf_batch_map_coordinates
The bilinear interpolation operation corresponding to the tf_batch_map_coordinates function:
Get the position of 4 coordinate points around the sampling position
Get the pixel value of the sampling position, bilinear interpolation to get the actual sampling result
'''

def tf_batch_map_coordinates(input, coords):
    '''
    Parameters
    ----------
    input : tf.Tensor. shape = (b*c, L, h, w)
    coords : tf.Tensor. shape = (b*c,l*h*w,3)
    Returns
    -------
    tf.Tensor. shape = (b*c, L, h, w)
    '''
    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    n = tf.argmax(input_shape)
    input_size = input_shape[n]
    n_coords = tf.shape(coords)[1]  # l*h*w

    #

    # Tensor("conv1_1/strided_slice_8:0", shape=(), dtype=int32)

    '''
    tf.clip_by_value(A, min, max)：
    Enter a tensor A to compress the value of each element in A between min and max.
    '''
    coords = tf.clip_by_value(coords, 0, tf.cast(input_size, 'float32') - 1)  # b*c,l*h*w,3

    # Get the integer coordinates of the upper left corner of the target coordinates (left top)
    coords_000 = tf.cast(tf.floor(coords), 'int32')
    # Get the integer coordinates of the lower right corner
    coords_111 = tf.cast(tf.math.ceil(coords), 'int32')
    # Get the three coordinates of the same level as the upper left corner
    coords_010 = tf.stack([coords_000[..., 0],
                           coords_111[..., 1],
                           coords_000[..., 2]], axis=-1)
    coords_100 = tf.stack([coords_111[..., 0],
                           coords_000[..., 1],
                           coords_000[..., 2]], axis=-1)
    coords_110 = tf.stack([coords_111[..., 0],
                           coords_111[..., 1],
                           coords_000[..., 2]], axis=-1)
    # Get the three coordinates of the same level as the lower right corner
    coords_001 = tf.stack([coords_000[..., 0],
                           coords_000[..., 1],
                           coords_111[..., 2]], axis=-1)
    coords_101 = tf.stack([coords_111[..., 0],
                           coords_000[..., 1],
                           coords_111[..., 2]], axis=-1)
    coords_011 = tf.stack([coords_000[..., 0],
                           coords_111[..., 1],
                           coords_111[..., 2]], axis=-1)

    # b*c is 5, h*w is 4, the total is the total of all coordinates of all pictures

    def _get_vals_by_coords(input, coords, batch_size, n_coords):
        idx = tf_repeat(tf.range(batch_size), n_coords)  # b*c*L*h*w
        indices = tf.stack([idx,
                            tf_flatten(coords[..., 0]),
                            tf_flatten(coords[..., 1]),
                            tf_flatten(coords[..., 2])], axis=-1)  # (b*c*L*h*w, 3)

        vals = tf.gather_nd(input, indices)  #Get the value from the indices
        vals = tf.reshape(vals, (batch_size, n_coords))  # (b*c,L*h*w)
        return vals

    # Get the pixel values of eight points based off  co-ordinates calculated
    vals000 = _get_vals_by_coords(input, coords_000, batch_size, n_coords)
    vals010 = _get_vals_by_coords(input, coords_010, batch_size, n_coords)
    vals100 = _get_vals_by_coords(input, coords_100, batch_size, n_coords)
    vals110 = _get_vals_by_coords(input, coords_110, batch_size, n_coords)

    vals001 = _get_vals_by_coords(input, coords_001, batch_size, n_coords)
    vals101 = _get_vals_by_coords(input, coords_101, batch_size, n_coords)
    vals011 = _get_vals_by_coords(input, coords_011, batch_size, n_coords)
    vals111 = _get_vals_by_coords(input, coords_111, batch_size, n_coords)

    '''
     Use bilinear interpolation on the 000 level to get a value mapped_vals_000
     Its coordinates are (coords_offset_000[..., 0],coords_offset_000[..., 1],0) + coords_000
    '''
    coords_offset_000 = coords - tf.cast(coords_000, 'float32')
    vals_t000 = vals000 + (vals100 - vals000) * coords_offset_000[..., 0]
    vals_b000 = vals010 + (vals110 - vals010) * coords_offset_000[..., 0]
    mapped_val_000 = vals_t000 + (vals_b000 - vals_t000) * coords_offset_000[..., 1]

    '''
    111 horizontal plane uses bilinear interpolation to get a value mapped_vals_111
    Its coordinates are (coords_offset_000[..., 0],coords_offset_000[..., 1],0) + coords_001
    '''
    vals_t000 = vals001 + (vals101 - vals001) * coords_offset_000[..., 0]
    vals_b000 = vals011 + (vals111 - vals011) * coords_offset_000[..., 0]
    mapped_vals_111 = vals_t000 + (vals_b000 - vals_t000) * coords_offset_000[..., 1]

    '''
    Use mapped_val_000 and mapped_vals_111 to get the final interpolation mapped_vals. Its coordinates are coords_00
    '''
    mapped_vals = mapped_val_000 + (mapped_vals_111 - mapped_val_000) * coords_offset_000[..., 2]

    return mapped_vals


def tf_batch_map_offsets(input, offsets):
    """
    Parameters
    ----------
    input : tf.Tensor. shape = (b*c, L, h, w)
    offsets : tf.Tensor. shape = (b*c, L, h, w,3)
    Returns
    -------
    tf.Tensor. shape = (b*c, L, h, w)
    """

    #  input.shape,offsets.shape: (?, 32, 32, 32)  (?, 32, 32, 32, 3)

    input_shape = tf.shape(input)  # (b*c,l,h,w)
    batch_size = input_shape[0]

    offsets = tf.reshape(offsets, (batch_size, -1, 3))  # (b*c,l*h*w,3)
    grid = tf.meshgrid(tf.range(input_shape[1]),
                       tf.range(input_shape[2]),
                       tf.range(input_shape[3]))
    grid = tf.stack(grid, axis=-1)  # (l,h,w,3)

    grid = tf.cast(grid, 'float32')
    grid = tf.reshape(grid, (-1, 3))  # (l*h*w,3)

    grid = tf_repeat_2d(grid, batch_size)  # (b*c,l*h*w,3)
    coords = offsets + grid

    mapped_vals = tf_batch_map_coordinates(input, coords)
    return mapped_vals