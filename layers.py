import tensorflow as tf
import numpy as np
import math
from libs.activations import lrelu


# useful macros
# variables
def weight_variable( name, shape, stddev=0.01 ):
  initial = tf.truncated_normal(shape, stddev=stddev)
  return tf.Variable(initial, name=name )

def bias_variable( name, shape ):
  initial = tf.constant(0.01, shape=shape )
  return tf.Variable(initial, name=name )

def resnet_relu( input ):
  return lrelu( input )

def batch_norm( input, phase, params, scope ):
  return tf.contrib.layers.batch_norm( input,
                                       center=params['batch_norm_center'],
                                       scale=params['batch_norm_scale'],
                                       decay=params['batch_norm_decay'],
                                       zero_debias_moving_mean=params['batch_norm_zero_debias'],
                                       is_training=phase,
                                       scope = scope )

def bn_dense( input, size_in, size_out, phase, params, scope ):
  stddev = np.sqrt( 2.0 / np.float32( size_in + size_out ))
  W = weight_variable( 'W', [ size_in, size_out ], stddev )
  b = bias_variable( 'bias', [ size_out ] )
  bn = batch_norm( input, phase, params, scope )
  return resnet_relu( tf.matmul( bn, W ) + b )

def gaussian_kernel(size: int, mean, std):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

    gauss_kernel = tf.einsum('i,j->ij', vals, vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)


# moved to simpler resnet layers, only one conv, no relu after identity

class layer_conv3d:
  def __init__( self, layer_id, variables, input, phase, params ):
    with tf.variable_scope( layer_id ):

      self.input_shape = input.shape.as_list()
      self.output_shape = variables.encoder_W.shape.as_list()

      # if output and input depth differ, we connect them via a 1x1 kernel embedding/projection
      # transformation.
      if variables.resample:
        identity = tf.nn.conv3d( input, variables.encoder_embedding, strides=variables.stride, padding='SAME' )
      else:
        identity = input
        
      #self.out = conv3d_batchnorm_relu( input, self.W, self.b, phase )
      # this time, BN on input
      self.bn   = batch_norm( input, phase, params, 'batchnorm_input' )
      self.conv     = tf.nn.conv3d( self.bn, variables.encoder_W, strides=variables.stride, padding='SAME' )
      self.features = resnet_relu( self.conv + variables.encoder_b )
      
      self.out   = self.features + identity
      self.output_shape = self.out.shape.as_list()


class layer_conv_one:
    def __init__(self, layer_id, variables, input):
        with tf.variable_scope(layer_id):

            self.conv = tf.nn.conv2d(input, variables.concat_W, strides=[1,1,1,1], padding='SAME')
            self.features = resnet_relu(self.conv + variables.concat_b)


            self.out = self.features
            self.output_shape = self.out.shape.as_list()


class layer_concat:
  def __init__(self, input1, input2):

      concat_v = input1[:, 0, :, :, :]
      concat_h = input2[:, 0, :, :, :]
      sh = input1.shape.as_list()[1]
      for view in range(1, sh):
        temp_v = input1[:, view, :, :, :]
        temp_h = input2[:, view, :, :, :]
        concat_v = tf.concat([concat_v, temp_v], 3)
        concat_h = tf.concat([concat_h, temp_h], 3)

      concat_vh = tf.concat([concat_v, tf.transpose(concat_h, [0, 2, 1, 3])], 3)

      self.out = concat_vh
      self.output_shape = self.out.shape.as_list()


class layer_pure_conv2D:
    def __init__(self, layer_id, layout, input, phase, params, no_relu = False):
        with tf.variable_scope(layer_id):
            self.shape = layout['conv']
            self.stride = layout['stride']
            self.input_shape = input.shape.as_list()
            self.C_in = self.shape[-2]
            self.C_out = self.shape[-1]

            self.n = self.shape[0] * self.shape[1] * self.shape[2] + self.shape[3]
            self.stddev = np.sqrt(2.0 / self.n)
            self.encoder_W = weight_variable('encoder_W', self.shape, self.stddev)
            self.encoder_b = bias_variable('encoder_b', [self.shape[3]])
            # self.out = conv3d_batchnorm_relu( input, self.W, self.b, phase )
            # this time, BN on input
            self.bn = batch_norm(input, phase, params, 'batchnorm_input')
            self.conv = tf.nn.conv2d(self.bn, self.encoder_W, strides=self.stride, padding='SAME')
            if no_relu:
                self.features = self.conv + self.encoder_b
                print('meow: no relu')
            else:
                self.features = tf.nn.relu(self.conv + self.encoder_b)


            self.out = self.features
            self.output_shape = self.out.shape.as_list()

class layer_strided_conv2D:
    def __init__(self, layer_id, input,channel):
        with tf.variable_scope(layer_id):

            self.shape = [3,3,channel,channel]
            self.stride = [1,2,2,1]
            self.input_shape = input.shape.as_list()
            self.C_in = self.shape[-2]
            self.C_out = self.shape[-1]

            self.n = self.shape[0] * self.shape[1] * self.shape[2] + self.shape[3]
            self.stddev = np.sqrt(2.0 / self.n)
            self.encoder_W = weight_variable('encoder_W', self.shape, self.stddev)
            # self.encoder_b = bias_variable('encoder_b', [self.shape[3]])
            # self.out = conv3d_batchnorm_relu( input, self.W, self.b, phase )
            # this time, BN on input
            # self.bn = batch_norm(input, phase, params, 'batchnorm_input')
            self.conv = tf.nn.conv2d(input, self.encoder_W, strides=self.stride, padding='SAME')
            # if no_relu:
            #     self.features = self.conv + self.encoder_b
            #     print('meow: no relu')
            # else:
            #     self.features = tf.nn.relu(self.conv + self.encoder_b)


            self.out = self.conv
            # self.output_shape = self.out.shape.as_list()


class layer_pure_deconv2D:
    def __init__(self, layer_id, layout, batch_size, input, phase, params):
        with tf.variable_scope(layer_id):

            self.shape = layout['conv']
            self.stride = layout['stride']
            self.input_shape = input.shape.as_list()
            self.C_in = self.shape[-2]
            self.C_out = self.shape[-1]
            self.output_shape = input.shape.as_list()
            self.output_shape[0] = batch_size
            self.output_shape[1] = self.output_shape[1] * self.stride[1]
            self.output_shape[2] = self.output_shape[2] * self.stride[2]
            self.output_shape[3] = self.C_out

            # self.output_shape = variables.encoder_W.shape.as_list()
            # create decoder layer variables
            self.n = self.shape[0] * self.shape[1] * self.shape[2] + self.shape[3]
            self.stddev = np.sqrt(2.0 / self.n)
            self.encoder_W = weight_variable('encoder_W', self.shape, self.stddev)
            self.encoder_b = bias_variable('encoder_b', [self.shape[3]])

            # self.out = conv3d_batchnorm_relu( input, self.W, self.b, phase )
            # this time, BN on input
            self.bn = batch_norm(input, phase, params, 'batchnorm_input')
            self.conv = tf.nn.conv2d_transpose(self.bn,
                                               self.encoder_W,
                                               strides=self.stride,
                                               output_shape= self.output_shape,
                                               padding='SAME')
            self.features = resnet_relu(self.conv + self.encoder_b)


            self.out = self.features
            self.output_shape = self.out.shape.as_list()


def _upsample_along_axis(volume, axis, stride, mode='COPY'):

  shape = volume.get_shape().as_list()

  assert mode in ['COPY', 'ZEROS']
  assert 0 <= axis < len(shape)

  target_shape = shape[:]
  target_shape[axis] *= stride
  target_shape[0] = -1

  padding = tf.zeros(shape, dtype=volume.dtype) if mode == 'ZEROS' else volume
  parts = [volume] + [padding for _ in range(stride - 1)]
  volume = tf.concat(parts, min(axis+1, len(shape)-1))

  volume = tf.reshape(volume, target_shape)
  return volume



class layer_upconv3d:
  def __init__( self, layer_id, variables, batch_size, input, phase, params, shared_variables=True ):

    with tf.variable_scope( layer_id ):

      self.input_shape = input.shape.as_list()
      self.output_shape = variables.input_shape
      self.output_shape[0] = batch_size

      self.bn = batch_norm( input, phase, params, 'batchnorm_input' )

      if shared_variables:
        # main decoder pipe, variables are shared between horizontal and vertical
        self.conv = tf.nn.conv3d_transpose( self.bn, variables.decoder_W,
                                            output_shape=self.output_shape,
                                            strides=variables.stride, padding='SAME' )
        self.features = resnet_relu( self.conv + variables.decoder_b )

      else:

        # secondary decoder pipe (e.g. diffuse / specular), variables are shared between horizontal and vertical
        self.decoder_W = weight_variable( 'decoder_W', variables.shape, variables.stddev )
        self.decoder_b = bias_variable( 'decoder_b', [ variables.shape[ 3 ] ] )

        self.conv = tf.nn.conv3d_transpose( self.bn, self.decoder_W,
                                            output_shape=self.output_shape,
                                            strides=variables.stride, padding='SAME' )
        self.features = resnet_relu( self.conv + self.decoder_b )


      if variables.resample:
        # bug (?) workaround: conv3d_transpose with strides does not seem to work, no idea why.
        # instead, we use upsampling + conv3d_transpose without stride.
        self.input_upsampled = input
        if variables.stride[1] != 1:
          self.input_upsampled = _upsample_along_axis( self.input_upsampled, 1, variables.stride[1] )
        if variables.stride[2] != 1:
          self.input_upsampled = _upsample_along_axis( self.input_upsampled, 2, variables.stride[2] )
        if variables.stride[3] != 1:
          self.input_upsampled = _upsample_along_axis( self.input_upsampled, 3, variables.stride[3] )
        
        if self.input_upsampled.shape[1] != self.output_shape[1]:
          # slightly hacky - crop if shape does not fit
          self.input_upsampled = self.input_upsampled[ :, 0:self.output_shape[1], :,:,: ]


        identity = tf.nn.conv3d_transpose( self.input_upsampled,
                                           variables.decoder_embedding,
                                           strides=[1,1,1,1,1],
                                           output_shape=self.output_shape,
                                           padding='SAME' )  

      else:
        identity = input

      self.out = self.features + identity


class layer_upconv2d_v2:
    def __init__(self, layer_id, variables, batch_size, input, phase, params, out_channels=-1, no_relu=False):

        with tf.variable_scope(layer_id):

            # define in/out shapes
            self.shape = variables.shape
            self.stride = variables.stride
            self.input_shape = input.shape.as_list()
            self.C_in = variables.C_in
            self.C_out = variables.C_out
            self.output_shape = input.shape.as_list()
            self.output_shape[0] = batch_size
            self.output_shape[1] = self.output_shape[1] * self.stride[1]
            self.output_shape[2] = self.output_shape[2] * self.stride[2]
            self.output_shape[3] = self.C_out
            self.W = variables.decoder_W
            self.b = variables.decoder_b


            self.resample = variables.resample
            if self.resample:
                self.embedding = variables.decoder_embedding

            if out_channels != -1:
                # output channel override
                self.output_shape[3] = out_channels
                self.shape[2] = out_channels
                self.C_in = out_channels
                self.resample = True

            # generate layers
            self.bn = batch_norm(input, phase, params, 'batchnorm_input')
            self.conv = tf.nn.conv2d_transpose(self.bn,
                                               self.W,
                                               output_shape=self.output_shape,
                                               strides=self.stride,
                                               padding='SAME')

            self.conv_rs = tf.reshape(self.conv, [-1] + self.output_shape[1:])
            if no_relu:
                self.features = self.conv_rs + self.b
            else:
                self.features = resnet_relu(self.conv_rs + self.b)

            if variables.resample:
                # bug (?) workaround: conv3d_transpose with strides does not seem to work, no idea why.
                # instead, we use upsampling + conv3d_transpose without stride.
                self.input_upsampled = input
                if self.stride[1] != 1:
                    self.input_upsampled = _upsample_along_axis(self.input_upsampled, 1, self.stride[1])
                if self.stride[2] != 1:
                    self.input_upsampled = _upsample_along_axis(self.input_upsampled, 2, self.stride[2])

                identity = tf.nn.conv2d(self.input_upsampled,
                                        self.embedding,
                                        strides=[1, 1, 1, 1],
                                        # output_shape=self.output_shape,
                                        padding='SAME')

            else:
                identity = input

            self.out = self.features + identity


# class layer_upconv2d_v2:
#     def __init__(self, layer_id, variables, batch_size, input, phase, params, out_channels=-1, no_relu=False):
#
#         with tf.variable_scope(layer_id):
#
#             # define in/out shapes
#             self.shape = variables.shape
#             self.stride = variables.stride
#             self.input_shape = input.shape.as_list()
#             self.C_in = variables.C_in
#             self.C_out = variables.C_out
#             self.output_shape = input.shape.as_list()
#             self.output_shape[0] = batch_size
#             if self.stride[1] > 1:
#                 if self.output_shape[1] == 3 or self.output_shape[1] == 5:
#                     self.output_shape[1] = self.output_shape[1] * self.stride[1] -1
#                     self.output_shape[2] = self.output_shape[2] * self.stride[2] -1
#                 else:
#                     self.output_shape[1] = self.output_shape[1] * self.stride[1]
#                     self.output_shape[2] = self.output_shape[2] * self.stride[2]
#
#             self.output_shape[3] = self.C_out
#             self.W = variables.decoder_W
#             self.b = variables.decoder_b
#
#
#             self.resample = variables.resample
#             if self.resample:
#                 self.embedding = variables.decoder_embedding
#
#             if out_channels != -1:
#                 # output channel override
#                 self.output_shape[3] = out_channels
#                 self.shape[2] = out_channels
#                 self.C_in = out_channels
#                 self.resample = True
#
#             # generate layers
#             self.bn = batch_norm(input, phase, params, 'batchnorm_input')
#             self.conv = tf.nn.conv2d_transpose(self.bn,
#                                                self.W,
#                                                output_shape=self.output_shape,
#                                                strides=self.stride,
#                                                padding='SAME')
#
#             self.conv_rs = tf.reshape(self.conv, [-1] + self.output_shape[1:])
#             if no_relu:
#                 self.features = self.conv_rs + self.b
#             else:
#                 self.features = resnet_relu(self.conv_rs + self.b)
#
#             if variables.resample:
#                 # bug (?) workaround: conv3d_transpose with strides does not seem to work, no idea why.
#                 # instead, we use upsampling + conv3d_transpose without stride.
#                 self.input_upsampled = input
#                 if self.stride[1] != 1 or self.stride[2] != 1:
#                     self.input_upsampled = tf.pad(self.input_upsampled,
#                                                   [[0, 0], [0, self.output_shape[1] - self.input_shape[1]],
#                                                    [0, self.output_shape[1] - self.input_shape[1]], [0, 0]])
#                 # if self.stride[1] != 1:
#                 #     self.input_upsampled = _upsample_along_axis(self.input_upsampled, 1, self.stride[1])
#                 # if self.stride[2] != 1:
#                 #     self.input_upsampled = _upsample_along_axis(self.input_upsampled, 2, self.stride[2])
#
#                 identity = tf.nn.conv2d(self.input_upsampled,
#                                         self.embedding,
#                                         strides=[1, 1, 1, 1],
#                                         # output_shape=self.output_shape,
#                                         padding='SAME')
#
#             else:
#                 identity = input
#
#             self.out = self.features + identity




      
class encoder_variables:
  def __init__( self, layer_id, layout ):

    # define variables for standard resnet layer for both conv as well as upconv
    with tf.variable_scope( layer_id ):
      self.shape  = layout[ 'conv' ]
      self.stride = layout[ 'stride' ]
      # to be initialized when building the conv layers
      self.input_shape  = []
      self.output_shape = []
      
      # number of channels in/out -> determines need for identity remapping
      self.C_in  = self.shape[-2]
      self.C_out = self.shape[-1]
      self.resample = self.C_in != self.C_out or self.stride[1] != 1 or self.stride[2] != 1 or self.stride[3] != 1
      if self.resample:
        self.project_n = self.C_in * self.stride[1] * self.stride[2] * self.stride[3] + self.C_out
        self.project_stddev = np.sqrt( 2.0 / self.project_n )
        self.encoder_embedding = weight_variable( 'encoder_embedding', [ 1,1,1, self.C_in, self.C_out ], stddev=self.project_stddev )

        
      # number of connections of a channel
      self.n = self.shape[0] * self.shape[1] * self.shape[2] * self.shape[3] + self.shape[4]
      self.stddev = np.sqrt( 2.0 / self.n )
      
      self.encoder_W = weight_variable( 'encoder_W', self.shape, self.stddev )
      self.encoder_b = bias_variable( 'encoder_b', [ self.shape[ 4 ] ] )


# class decoder_variables_2D:
#     def __init__(self, layer_id, layout, i, last_layer, patch_weight, nviews):
#         # define variables for standard resnet layer for both conv as well as upconv
#         with tf.variable_scope(layer_id):
#             if i == last_layer-1:
#                 self.shape = [layout['conv'][1], layout['conv'][2],
#                               int(layout['conv'][3] * patch_weight),int(layout['conv'][4]* patch_weight)]
#             else:
#                 self.shape = [layout['conv'][1], layout['conv'][2],
#                               int(layout['conv'][3] * patch_weight),
#                               int(layout['conv'][4] * (patch_weight + 1))]
#             self.stride = [layout['stride'][0], layout['stride'][2], layout['stride'][3], layout['stride'][4]]
#             # to be initialized when building the conv layers
#             self.input_shape = []
#             self.output_shape = []
#
#             # number of channels in/out -> determines need for identity remapping
#             self.C_in = self.shape[-1]
#             self.C_out = self.shape[-2]
#             self.resample = self.C_in != self.C_out or self.stride[1] != 1 or self.stride[2] != 1
#             if self.resample:
#                 self.project_n = self.C_in * self.stride[1] * self.stride[2]  + self.C_out
#                 self.project_stddev = np.sqrt(2.0 / self.project_n)
#                 self.decoder_embedding = weight_variable('decoder_embedding', [1, 1, self.C_in, self.C_out],
#                                                          stddev=self.project_stddev)
#
#             # number of connections of a channel
#             self.n = self.shape[0] * self.shape[1] * self.shape[2]  + self.shape[3]
#             self.stddev = np.sqrt(2.0 / self.n)
#
#             self.decoder_W = weight_variable('decoder_W', self.shape, self.stddev)
#             self.decoder_b = bias_variable('decoder_b', [self.shape[2]])


class decoder_variables_2D:
    def __init__(self, layer_id, layout, i, last_layer, patch_weight, skip_connection):
        # define variables for standard resnet layer for both conv as well as upconv
        with tf.variable_scope(layer_id):
            if i == last_layer-1:
                self.shape = [layout['conv'][1], layout['conv'][2],
                              int(layout['conv'][3] * patch_weight),int(layout['conv'][4]* patch_weight)]
            else:
                if skip_connection:
                    self.shape = [layout['conv'][1], layout['conv'][2],
                                  int(layout['conv'][3] * patch_weight),
                                  int(layout['conv'][4] * (patch_weight + 1))]
                else:
                    self.shape = [layout['conv'][1], layout['conv'][2],
                                  int(layout['conv'][3] * patch_weight),
                                  int(layout['conv'][4] * patch_weight)]

            self.stride = [layout['stride'][0], layout['stride'][2], layout['stride'][3], layout['stride'][4]]
            # to be initialized when building the conv layers
            self.input_shape = []
            self.output_shape = []

            # number of channels in/out -> determines need for identity remapping
            self.C_in = self.shape[-1]
            self.C_out = self.shape[-2]
            self.resample = self.C_in != self.C_out or self.stride[1] != 1 or self.stride[2] != 1
            if self.resample:
                self.project_n = self.C_in * self.stride[1] * self.stride[2]  + self.C_out
                self.project_stddev = np.sqrt(2.0 / self.project_n)
                self.decoder_embedding = weight_variable('decoder_embedding', [1, 1, self.C_in, self.C_out],
                                                         stddev=self.project_stddev)

            # number of connections of a channel
            self.n = self.shape[0] * self.shape[1] * self.shape[2]  + self.shape[3]
            self.stddev = np.sqrt(2.0 / self.n)

            self.decoder_W = weight_variable('decoder_W', self.shape, self.stddev)
            self.decoder_b = bias_variable('decoder_b', [self.shape[2]])

# class concat_variables:
#     def __init__(self, layer_id, layout, patches, input_features, output_features):
#         # define variables for standard resnet layer for both conv as well as upconv
#         with tf.variable_scope(layer_id):
#             if not input_features:
#                 self.shape = [1, 1,
#                               int(layout['conv'][4] * 2 * patches),
#                               layout['conv'][4]]
#             else:
#                 self.shape = [1, 1,
#                               int(input_features * 2 * patches),
#                               output_features]
#
#             # number of connections of a channel
#             self.n = self.shape[0] * self.shape[1] * self.shape[2]  + self.shape[3]
#             self.stddev = np.sqrt(2.0 / self.n)
#
#             self.concat_W = weight_variable('concat_W', self.shape, self.stddev)
#             self.concat_b = bias_variable('concat_b', [self.shape[3]])

class concat_variables:
    def __init__(self, layer_id, layout, patches, input_features, output_features, num_skip = 2):
        # define variables for standard resnet layer for both conv as well as upconv
        with tf.variable_scope(layer_id):
            if not input_features:
                self.shape = [1, 1,
                              int(layout['conv'][4] * num_skip * patches),
                              layout['conv'][4]]
            else:
                self.shape = [1, 1,
                              int(input_features * num_skip * patches),
                              output_features]

            # number of connections of a channel
            self.n = self.shape[0] * self.shape[1] * self.shape[2]  + self.shape[3]
            self.stddev = np.sqrt(2.0 / self.n)

            self.concat_W = weight_variable('concat_W', self.shape, self.stddev)
            self.concat_b = bias_variable('concat_b', [self.shape[3]])

# class upscale_variables:
#     def __init__(self, layer_id, layout):
#         # define variables for standard resnet layer for both conv as well as upconv
#         with tf.variable_scope(layer_id):
#             self.shape = layout['conv']
#             self.stride = layout['stride']
#             # to be initialized when building the conv layers
#             self.input_shape = []
#             self.output_shape = []
#
#             # number of channels in/out -> determines need for identity remapping
#             self.C_in = self.shape[-1]
#             self.C_out = self.shape[-2]
#             self.resample = self.C_in != self.C_out or self.stride[1] != 1 or self.stride[2] != 1
#             if self.resample:
#                 self.project_n = self.C_in * self.stride[1] * self.stride[2]  + self.C_out
#                 self.project_stddev = np.sqrt(2.0 / self.project_n)
#                 self.decoder_embedding = weight_variable('decoder_embedding', [1, 1, self.C_in, self.C_out],
#                                                          stddev=self.project_stddev)
#
#             # number of connections of a channel
#             self.n = self.shape[0] * self.shape[1] * self.shape[2]  + self.shape[3]
#             self.stddev = np.sqrt(2.0 / self.n)
#
#             self.decoder_W = weight_variable('decoder_W', self.shape, self.stddev)
#             self.decoder_b = bias_variable('decoder_b', [self.shape[2]])


class upscale_variables:
    def __init__(self, layer_id, layout, patch_weight, skip_connection, channels):
        # define variables for standard resnet layer for both conv as well as upconv
        with tf.variable_scope(layer_id):
            self.stride = layout['stride']
            # to be initialized when building the conv layers
            self.input_shape = []
            self.output_shape = []
            if skip_connection:
                self.shape = [layout['conv'][0], layout['conv'][1],
                              layout['conv'][2],int(layout['conv'][3]* (patch_weight + 1) )]
            else:
                self.shape = [layout['conv'][0], layout['conv'][1],
                              layout['conv'][2], layout['conv'][3]]

            # number of channels in/out -> determines need for identity remapping
            if layout['interp']==1:
                self.C_in = self.shape[-1] + channels
                self.shape[-1] = self.shape[-1] + channels
            else:
                self.C_in = self.shape[-1]

            self.C_out = self.shape[-2]
            self.resample = self.C_in != self.C_out or self.stride[1] != 1 or self.stride[2] != 1
            if self.resample:
                self.project_n = self.C_in * self.stride[1] * self.stride[2]  + self.C_out
                self.project_stddev = np.sqrt(2.0 / self.project_n)
                self.decoder_embedding = weight_variable('decoder_embedding', [1, 1, self.C_in, self.C_out],
                                                         stddev=self.project_stddev)

            # number of connections of a channel
            self.n = self.shape[0] * self.shape[1] * self.shape[2]  + self.shape[3]
            self.stddev = np.sqrt(2.0 / self.n)

            self.decoder_W = weight_variable('decoder_W', self.shape, self.stddev)
            self.decoder_b = bias_variable('decoder_b', [self.shape[2]])


# class upscale_variables:
#     def __init__(self, layer_id, layout, patch_weight, skip_connection):
#         # define variables for standard resnet layer for both conv as well as upconv
#         with tf.variable_scope(layer_id):
#             self.stride = layout['stride']
#             # to be initialized when building the conv layers
#             self.input_shape = []
#             self.output_shape = []
#             if skip_connection:
#                 self.shape = [layout['conv'][0], layout['conv'][1],
#                               int(layout['conv'][2] * patch_weight),int(layout['conv'][3]* (patch_weight + 1) )]
#             else:
#                 self.shape = [layout['conv'][0], layout['conv'][1],
#                               int(layout['conv'][2] * patch_weight),int(layout['conv'][3]*patch_weight )]
#
#             # number of channels in/out -> determines need for identity remapping
#             self.C_in = self.shape[-1]
#             self.C_out = self.shape[-2]
#             self.resample = self.C_in != self.C_out or self.stride[1] != 1 or self.stride[2] != 1
#             if self.resample:
#                 self.project_n = self.C_in * self.stride[1] * self.stride[2]  + self.C_out
#                 self.project_stddev = np.sqrt(2.0 / self.project_n)
#                 self.decoder_embedding = weight_variable('decoder_embedding', [1, 1, self.C_in, self.C_out],
#                                                          stddev=self.project_stddev)
#
#             # number of connections of a channel
#             self.n = self.shape[0] * self.shape[1] * self.shape[2]  + self.shape[3]
#             self.stddev = np.sqrt(2.0 / self.n)
#
#             self.decoder_W = weight_variable('decoder_W', self.shape, self.stddev)
#             self.decoder_b = bias_variable('decoder_b', [self.shape[2]])



class decoder_variables:
  def __init__( self, layer_id, encoder_variables ):

    # define variables for standard resnet layer for both conv as well as upconv
    with tf.variable_scope( layer_id ):
      self.shape  = encoder_variables.shape
      self.stride = encoder_variables.stride
      self.input_shape  = encoder_variables.input_shape
      self.output_shape = encoder_variables.output_shape
      self.C_in  = encoder_variables.C_in
      self.C_out = encoder_variables.C_out
      self.resample = encoder_variables.resample

      if self.resample:
        self.project_n = self.C_in * self.stride[1] * self.stride[2] * self.stride[3] + self.C_out
        self.project_stddev = np.sqrt( 2.0 / self.project_n )
        self.decoder_embedding = weight_variable( 'decoder_embedding', [ 1,1, self.C_in, self.C_out ], stddev=self.project_stddev )

      # number of connections of a channel
      self.n = self.shape[0] * self.shape[1] * self.shape[2] * self.shape[3] + self.shape[4]
      self.stddev = np.sqrt( 2.0 / self.n )
      
      self.decoder_W = weight_variable( 'decoder_W', self.shape, self.stddev )
      self.decoder_b = bias_variable( 'decoder_b', [ self.shape[ 3 ] ] )
      

def pinhole_conv3d(variables, input):

  conv = tf.nn.conv3d( input, variables.pinhole_weight, strides=[1,1,1,1,1], padding='SAME' )
  return conv

def pinhole_weight(variables, input):
  shape = input.shape.as_list()

  pinhole_weight = weight_variable('pinhole_weight', [1, 1, 1, shape[-1], variables.C_in],
                                   stddev = variables.stddev)
  return pinhole_weight

def pinhole_conv2d(variables, input):

  conv = tf.nn.conv2d( input, variables.pinhole_weight, strides=[1,1,1,1], padding='SAME' )
  return conv

def pinhole_weight_2d(variables, input):
  shape = input.shape.as_list()

  pinhole_weight = weight_variable('pinhole_weight', [1, 1, shape[-1], variables.C_in],
                                   stddev = variables.stddev)
  return pinhole_weight