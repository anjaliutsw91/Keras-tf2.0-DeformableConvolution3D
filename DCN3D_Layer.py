
# -*- coding: utf-8 -*-
"""
Created on Sunday June 12 16:33:21 2021

Author: Anjali Balagopal
Email: anjalibalagopal91@gmail.com

Code adapted from : https://www.programmersought.com/article/5408456601/ & https://github.com/wcfzl/3D-CNNs-for-Liver-Classification.git
"""


from Utils.Models.DeformableConvolution3D import tf_batch_map_offsets
import tensorflow as tf
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.initializers import RandomNormal


class ConvOffset3D(Conv3D):
    """
    ConvOffset3D
    """

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init"""

        self.filters = filters
        super(ConvOffset3D, self).__init__(self.filters * 3, (3, 3, 3),
                                           padding='same',
                                           use_bias=False,
                                           kernel_initializer=RandomNormal(0, init_normal_stddev),
                                           **kwargs)

    def call(self, x):
        """Return the deformed featured map"""

        x_shape = x.get_shape()
        offsets = super(ConvOffset3D, self).call(x)
        # offsets: (b*c,l,h,w,3)
        offsets = self._to_bc_L_h_w_3(offsets, x_shape)
        x = self._to_bc_L_h_w(x, x_shape)
        x_offset = tf_batch_map_offsets(x, offsets)
        x_offset = self._to_b_c_L_h_w(x_offset, x_shape)
        return x_offset

    def compute_output_shape(self, input_shape):
        """Output shape is the same as input shape
        Because this layer does only the deformation part
        """
        return input_shape

    @staticmethod
    def _to_bc_L_h_w_3(x, x_shape):
        """(b, h, w, 2c) -> (b*c, h, w, 2)  ##  (b, L, h, w, 2c) -> (b*c, L, h, w, 2)"""

        x = tf.reshape(x, (-1,
                           int(x_shape[2]),
                           int(x_shape[3]),
                           int(x_shape[4]), 3))
        return x

    @staticmethod
    def _to_bc_L_h_w(x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)  ##  (b, L, h, w, c) -> (b*c, L, h, w)"""
        x = tf.reshape(x, (-1,
                           int(x_shape[2]),
                           int(x_shape[3]),
                           int(x_shape[4])))
        return x

    @staticmethod
    def _to_b_c_L_h_w(x, x_shape):
        """##  (b*c, L*h*w) -> (b, c, L, h, w)"""
        x = tf.reshape(x, (-1,
                           int(x_shape[1]),
                           int(x_shape[2]),
                           int(x_shape[3]),
                           int(x_shape[4])))
        return x
