# Class definition for the combined CRY network
# drops: deep regression on angular patch stacks
#
# in this version, we take great care to have nice
# variable scope names.
#
# start session
import code
import tensorflow as tf
import numpy as np
import math
import libs.layers as layers
from tensorflow.image import yuv_to_rgb, resize_bicubic

from resnet_v1 import resnet_v1, resnet_arg_scope, resnet_v1_50
from inception.inception_v3 import inception_v3_arg_scope, inception_v3
from vgg19 import vgg_19
# from mobilenet.mobilenet_v2 import mobilenet

# from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.image import resize_bilinear
import tensorflow.contrib.slim as slim
from refocus_depth import refocus_custom_tf1,bilinear_interp_3d
from tensorflow.image import resize_area



# main class defined in module
class create_cnn:

  def __init__( self, config ):

    # config (hyperparameters)
    self.config = config
    self.max_layer = config.config[ 'max_layer' ]
    self.interpolate = config.config['interpolate']
    # we get two input paths for autoencoding:
    # 1. vertical epi stack in stack_v
    # 2. horizontal epi stack in stack_h

    # both stacks have 9 views, patch size 16x16 + 16 overlap on all sides,
    # for a total of 48x48.
    self.C = config.C
    self.C_value = config.C_value
    self.cv_pos = config.cv_pos
    self.D = config.D
    self.D_in = config.D_in
    self.H = config.H
    self.W = config.W
    self.H_s2 = config.H_s2
    self.W_s2 = config.W_s2
    self.H_s4 = config.H_s4
    self.W_s4 = config.W_s4
    self.reuse_resnet = False
    self.reuse_vgg = False
    self.reuse_mobilenet = False
    self.reuse_inception = False

    # regularization weights
    self.beta = 0.0001
    self.loss_min_coord_3D = dict()
    self.loss_max_coord_3D = dict()

    self.loss_min_coord_3D['s2'] = np.int(config.sx_s2*0.25) # 0
    self.loss_max_coord_3D['s2'] = np.int(config.W_s2 - config.sx_s2 * 0.25) # config.W_s2
    self.loss_min_coord_3D['s4'] = np.int(config.sx_HR*0.25) # 0 #
    self.loss_max_coord_3D['s4'] = np.int(config.W_s4 - config.sx_HR * 0.25) # config.W_s4 #
    self.scales = []
    # input layers
    with tf.device( '/device:GPU:%i' % ( self.config.layers['preferred_gpu'] ) ):
      with tf.variable_scope( 'input' ):

        self.stack_v = tf.placeholder(tf.float32, shape=[None, self.D_in, self.H, self.W, self.C_value] )
        self.stack_h = tf.placeholder(tf.float32, shape=[None, self.D_in, self.H, self.W, self.C_value] )

        self.stack_shape = self.stack_v.shape.as_list()
        self.stack_shape[ 0 ] = -1

        self.phase = tf.placeholder(tf.bool, name='phase')
        self.keep_prob = tf.placeholder(tf.float32)
        self.noise_sigma = tf.placeholder(tf.float32)

    # FEATURE LAYERS
    self.batch_size = tf.shape(self.stack_v)[0]

    self.encoders_3D = dict()
    self.decoders_3D = dict()
    self.minimizers = dict()

    self.create_3D_encoders()
    self.create_3D_decoders()
    if len(config.discriminator) > 0:
        self.discriminator_config = config.discriminator[0]
        self.create_discriminator()
        self.use_gan = True
    else:
        self.use_gan = False
    self.setup_losses()

#
   # CREATE DECODER LAYERS FOR ADDITIONAL DECODERS CONFIGURED IN THE CONFIG FILE
   #
  def create_3D_encoders(self):
      for encoder_config in self.config.encoders_3D:
          with tf.device('/device:GPU:%i' % (encoder_config['preferred_gpu'])):
              self.create_3D_encoder(encoder_config)

  def create_3D_encoder(self, encoder_config):
      encoder = dict()
      encoder_id = encoder_config['id']
      layout = []
      for i in range(0,len(self.config.layers['encoder_3D'])):
          layout.append(self.config.layers['encoder_3D'][i])
      print('creating encoder pipeline for ' + encoder_id)
      with tf.variable_scope(encoder_id):
          encoder['id'] = encoder_id
          encoder['channels'] = encoder_config['channels']
          encoder['preferred_gpu'] = encoder_config['preferred_gpu']
          encoder['variables'] = []
          encoder['features_v'] = None
          encoder['features_h'] = None
          encoder['conv_layers_v'] = []
          encoder['conv_layers_h'] = []
          ####################################################################################################
          # create encoder variables
          last_layer = min(len(layout), self.max_layer)
          for i in range(0, last_layer):
              layer_id = "encoder_%i" % i
              print('    creating 3D encoder variables ' + layer_id)
              encoder['variables'].append(layers.encoder_variables(layer_id, layout[i]))
          ####################################################################################################
          # create 3D encoder layers for stacks
          shape = [self.stack_shape[0],self.stack_shape[1],self.stack_shape[2],
                   self.stack_shape[3],self.stack_shape[4]]
          if encoder['features_v'] == None:
              encoder['features_v'] = self.stack_v
              # if encoder['channels'] != self.config.layer_config[pos]['layout'][0]['conv'][-2]:
              #     encoder['features_v'] = tf.concat([encoder['features_v'],self.stack_v[:,:,:,:,3:]], axis = -1 )
              encoder['features_v'] = tf.reshape(encoder['features_v'], shape) # why we need to reshape ?
              # encoder['features_v_input'] = encoder['features_v']
          if encoder['features_h'] == None:
              encoder['features_h'] = self.stack_h
              # if encoder['channels'] != self.config.layer_config[pos]['layout'][0]['conv'][-2]:
              #     encoder['features_h'] = tf.concat([encoder['features_h'],self.stack_h[:,:,:,:,3:]], axis = -1 )
              encoder['features_h'] = tf.reshape(encoder['features_h'], shape)
              # encoder['features_h_input'] = encoder['features_h']
          print('    CREATING 3D encoder layers for %s ' % encoder_id)
          for i in range(0, last_layer):
              layer_id_v = "v_%s_%i" % (encoder_id,i)
              layer_id_h = "h_%s_%i" % (encoder_id,i)
              print('    generating downconvolution layer structure for %s %i' % (encoder_id,i))
              encoder['conv_layers_v'].append(layers.layer_conv3d(layer_id_v,
                                                                  encoder['variables'][i],
                                                                  encoder['features_v'],
                                                                  self.phase,
                                                                  self.config.training))
              encoder['conv_layers_h'].append(layers.layer_conv3d(layer_id_h,
                                                                  encoder['variables'][i],
                                                                  encoder['features_h'],
                                                                  self.phase,
                                                                  self.config.training))
              # update layer shapes
              encoder['variables'][i].input_shape = encoder['conv_layers_v'][i].input_shape
              encoder['variables'][i].output_shape = encoder['conv_layers_v'][i].output_shape
              # final encoder layer: vertical/horizontal features
              encoder['features_v'] = encoder['conv_layers_v'][-1].out
              encoder['features_h'] = encoder['conv_layers_h'][-1].out
          ####################################################################################################
          # create dense layers

          self.encoders_3D[encoder_id] = encoder
  #
  # CREATE DECODER LAYERS FOR ADDITIONAL DECODERS CONFIGURED IN THE CONFIG FILE
  #
  def create_3D_decoders( self ):
    for decoder_config in self.config.decoders_3D:
        self.create_3D_decoder( decoder_config)


  def create_3D_decoder( self, decoder_config):

    decoder = dict()
    decoder_id = decoder_config[ 'id' ]
    ids = []
    for i in range(0, len(self.config.layer_config)):
        ids.append(self.config.layer_config[i]['id'])
    pos_layout = ids.index(decoder_id)
    print( 'creating decoder pipeline ' + decoder_id )
    self.id = decoder_id

    # create a decoder pathway (center view only)
    with tf.variable_scope( decoder_id ):

        with tf.device('/device:GPU:%i' % (decoder_config['preferred_gpu'][0])):

          decoder[ 'id' ] = decoder_id
          decoder[ 'channels' ] = decoder_config[ 'channels' ]
          decoder[ 'loss_fn' ] = decoder_config[ 'loss_fn' ]
          decoder[ 'weight' ] = decoder_config[ 'weight' ]
          decoder[ 'percep_loss_weight'] = decoder_config['percep_loss_weight']
          decoder[ 'train' ] = decoder_config[ 'train' ]
          decoder[ 'preferred_gpu' ] = decoder_config[ 'preferred_gpu' ]
          decoder[ 'start'] = self.config.layer_config[pos_layout]['start']
          decoder[ 'end'] = self.config.layer_config[pos_layout]['end']
          decoder[ 'no_relu'] = decoder_config['no_relu']
          decoder[ 'skip_connection'] = decoder_config['skip_connection']
          decoder[ 'percep_loss'] = decoder_config['percep_loss']
          decoder['adv_loss_weight'] = decoder_config['adv_loss_weight']
          decoder['skip_id'] = decoder_config['skip_id']

          decoder['3D_variables'] = []
          decoder['upscale_variables'] = []



          decoder['upconv_v'] = self.encoders_3D[decoder_id]['features_v']
          decoder['upconv_h'] = self.encoders_3D[decoder_id]['features_h']

          decoder['layers_v'] = []
          decoder['layers_h'] = []

          ########################################################################################################
          # decoder variables
          layout = []
          for i in range(0, len(self.config.layers['encoder_3D'])):
              layout.append(self.config.layers['encoder_3D'][i])

          last_layer = min(len(layout), self.max_layer)

          layout[0] = self.config.layer_config[pos_layout]['upscale'][0]
          for i in range(0, last_layer):
              layer_id = "decoder_%s_%i" % (decoder_id, i)
              print('    generating upconvolution variables ' + layer_id)
              decoder['3D_variables'].append(layers.decoder_variables_3D(layer_id, layout[i],
                                                                         i, last_layer, self.config.patch_weight,
                                                                         decoder['skip_connection']))

          for i in range(0, last_layer):
              layer_id_v = "decoder_v_%s_layer%i" % (decoder_id, last_layer - i - 1)
              layer_id_h = "decoder_h_%s_layer%i" % (decoder_id, last_layer - i - 1)
              print('    generating upconvolution layer structure ' + layer_id_v)
              if i != last_layer - 1:
                  output_shape = self.encoders_3D[decoder_id]['conv_layers_v'][-2 - i].out.shape.as_list()[2:4]
              else:
                  output_shape = [self.H, self.W]

              decoder['layers_v'].insert(-1 - i,
                                       layers.layer_decoder_upconv3d(layer_id_v,
                                                                decoder['3D_variables'][-1 - i],
                                                                self.batch_size,
                                                                output_shape,
                                                                decoder['upconv_v'],
                                                                self.phase,
                                                                self.config.training))
              print('    generating upconvolution layer structure ' + layer_id_h)
              decoder['layers_h'].insert(-1 - i,
                                         layers.layer_decoder_upconv3d(layer_id_h,
                                                                       decoder['3D_variables'][-1 - i],
                                                                       self.batch_size,
                                                                       output_shape,
                                                                       decoder['upconv_h'],
                                                                       self.phase,
                                                                       self.config.training))

              if decoder['skip_connection']:
                  if i != last_layer - 1:
                      skip_v = self.encoders_3D[decoder['skip_id'][0]]['conv_layers_v'][-2 - i].out
                      skip_h = self.encoders_3D[decoder['skip_id'][0]]['conv_layers_h'][-2 - i].out
                      sh_tmp = decoder['layers_v'][-1 - i].out.shape.as_list()
                      if  sh_tmp[1] == 5:
                          skip_v = layers._upsample_along_axis(skip_v, 1, 2)
                          skip_h = layers._upsample_along_axis(skip_h, 1, 2)
                      if sh_tmp[1] == 7 or sh_tmp[1] == 9:
                          skip_v = layers._upsample_along_axis(skip_v, 1, 3)
                          skip_h = layers._upsample_along_axis(skip_h, 1, 3)

                      if skip_v.shape[1] != sh_tmp[1]:
                              # slightly hacky - crop if shape does not fit
                          skip_v= skip_v[:, 0:sh_tmp[1], :, :, :]
                          skip_h = skip_h[:, 0:sh_tmp[1], :, :, :]

                  else:
                      decoder['stack_v'] = tf.placeholder(tf.float32,
                                                          [None, self.D, self.H, self.W, decoder['channels']])
                      decoder['stack_h'] = tf.placeholder(tf.float32,
                                                          [None, self.D, self.H, self.W, decoder['channels']])
                      skip_v = decoder['stack_v']
                      skip_h = decoder['stack_h']

                  decoder['upconv_v'] = tf.concat([decoder['layers_v'][-1 - i].out, skip_v], axis=-1)
                  decoder['upconv_h'] = tf.concat([decoder['layers_h'][-1 - i].out, skip_h], axis=-1)
                  decoder['3D_variables'][-1 - i].pinhole_weight = layers.pinhole_weight(decoder['3D_variables'][-1 - i],
                                                                                         decoder['upconv_v'])
                  decoder['upconv_v'] = layers.pinhole_conv3d(decoder['3D_variables'][-1-i], decoder['upconv_v'])
                  decoder['upconv_h'] = layers.pinhole_conv3d(decoder['3D_variables'][-1 - i], decoder['upconv_h'])
              else:
                  decoder['upconv_v'] = decoder['layers_v'][-1 - i].out
                  decoder['upconv_h'] = decoder['layers_h'][-1 - i].out

        with tf.device('/device:GPU:%i' % (decoder_config['preferred_gpu'][1])):
            layout_final_s2 = self.config.layer_config[pos_layout]['final_s2'][0]
            layout_final_s4 = self.config.layer_config[pos_layout]['final_s4'][0]
            # if self.interpolate:
            #     decoder['bicubic_h'] = tf.stack(
            #         [resize_bicubic(decoder['stack_h'][:, i, :, :, decoder['start']:decoder['end']], [self.H_s2, self.W_s2]) for i in
            #          range(0, self.D)], axis=1)
            #     decoder['bicubic_v'] = tf.stack(
            #         [resize_bicubic(decoder['stack_v'][:, i, :, :, decoder['start']:decoder['end']], [self.H_s2, self.W_s2]) for i in
            #          range(0, self.D)], axis=1)

            layout_upscale = []
            for i in range(0, len(self.config.layers['upscale'])):
                layout_upscale.append(self.config.layers['upscale'][i])

            last_layer_up = len(layout_upscale)
            interpolate = False
            for i in range(0, last_layer_up):
                # if i == 1:
                #     if self.interpolate:
                #         interpolate = True
                # else:
                #     interpolate = False
                layer_id = "upscale_%i" % i
                print('    creating 3D upscale variables ' + layer_id)
                decoder['upscale_variables'].append(layers.upscale_variables(layer_id, layout_upscale[i],
                                                                             interpolate, decoder['channels'], self.config.patch_weight))



            for i in range(0, last_layer_up):
                layer_id_v = "v_upscale_%s_%i" % (decoder_id, i)
                layer_id_h = "h_upscale_%s_%i" % (decoder_id, i)
                print('    generating upconvolution layer structure for %s %i' % (decoder_id, i))

                decoder['upconv_v'] = layers.layer_upscale_upconv3d(layer_id_v,
                                                                       decoder['upscale_variables'][i],
                                                                       self.batch_size,
                                                                       layout_upscale[i]['target_shape'],
                                                                       decoder['upconv_v'],
                                                                       self.phase,
                                                                       self.config.training).out
                decoder['upconv_h'] = layers.layer_upscale_upconv3d(layer_id_h,
                                                                    decoder['upscale_variables'][i],
                                                                    self.batch_size,
                                                                    layout_upscale[i]['target_shape'],
                                                                    decoder['upconv_h'],
                                                                    self.phase,
                                                                    self.config.training).out
                # if i == 0:
                #     decoder['upconv_v'] = tf.concat([decoder['upconv_v'], decoder['bicubic_v']],axis=-1)
                #     decoder['upconv_h'] = tf.concat([decoder['upconv_h'], decoder['bicubic_h']],axis=-1)

                if layout_upscale[i]['out'] == 's2':
                    no_relu = decoder['no_relu']
                    decoder['upconv_v_s2'] = layers.layer_pure_conv3D('decoder_v_final_s2', layout_final_s2,
                                                                      decoder['upconv_v'], self.phase,
                                                                         self.config.training, no_relu=no_relu).out
                    decoder['upconv_h_s2'] = layers.layer_pure_conv3D('decoder_h_final_s2', layout_final_s2,
                                                                      decoder['upconv_h'], self.phase,
                                                                         self.config.training, no_relu=no_relu).out
                    self.scales.append('s2')

                if layout_upscale[i]['out'] == 's4':
                    no_relu = decoder['no_relu']
                    decoder['upconv_v_s4'] = layers.layer_pure_conv3D('decoder_v_final_s4', layout_final_s4,
                                                                      decoder['upconv_v'], self.phase,
                                                                         self.config.training, no_relu=no_relu).out
                    decoder['upconv_h_s4'] = layers.layer_pure_conv3D('decoder_h_final_s4', layout_final_s4,
                                                                      decoder['upconv_h'], self.phase,
                                                                         self.config.training, no_relu=no_relu).out
                    self.scales.append('s4')

        for scale in self.scales:

            if scale == 's2':
                decoder['input_v_' + scale] = tf.placeholder(tf.float32, [None, self.D, self.H_s2, self.W_s2,
                                                                          decoder['channels']])
                decoder['input_h_' + scale] = tf.placeholder(tf.float32, [None, self.D, self.H_s2, self.W_s2,
                                                                          decoder['channels']])
            if scale == 's4':
                decoder['input_v_' + scale] = tf.placeholder(tf.float32, [None, self.D, self.H_s4, self.W_s4,
                                                                          decoder['channels']])
                decoder['input_h_' + scale] = tf.placeholder(tf.float32, [None, self.D, self.H_s4, self.W_s4,
                                                                          decoder['channels']])

            decoder['features_v_dd_' + scale] = tf.pad(decoder['upconv_v_' + scale],
                                                       [[0, 0], [0, 1], [0, 0], [0, 0], [0, 0]])[:, 1:, :,
                                                :, :] - decoder['upconv_v_' + scale]
            decoder['features_v_dx_' + scale] = tf.pad(decoder['upconv_v_' + scale],
                                                       [[0, 0], [0, 0], [0, 0], [0, 1], [0, 0]])[:, :, :,
                                                1:, :] - decoder['upconv_v_' + scale]
            decoder['features_v_dy_' + scale] = tf.pad(decoder['upconv_v_' + scale],
                                                       [[0, 0], [0, 0], [0, 1], [0, 0], [0, 0]])[:, :, 1:,
                                                :, :] - decoder['upconv_v_' + scale]
            decoder['features_h_dd_' + scale] = tf.pad(decoder['upconv_h_' + scale],
                                                       [[0, 0], [0, 1], [0, 0], [0, 0], [0, 0]])[:, 1:, :,
                                                :, :] - decoder['upconv_h_' + scale]
            decoder['features_h_dx_' + scale] = tf.pad(decoder['upconv_h_' + scale],
                                                       [[0, 0], [0, 0], [0, 0], [0, 1], [0, 0]])[:, :, :,
                                                1:, :] - decoder['upconv_h_' + scale]
            decoder['features_h_dy_' + scale] = tf.pad(decoder['upconv_h_' + scale],
                                                       [[0, 0], [0, 0], [0, 1], [0, 0], [0, 0]])[:, :, 1:,
                                                :, :] - decoder['upconv_h_' + scale]

            decoder['features_v_gt_dd_' + scale] = tf.pad(
                decoder['input_v_' + scale],
                [[0, 0], [0, 1], [0, 0], [0, 0], [0, 0]])[:, 1:, :,
                                                   :, :] - decoder[
                                                       'input_v_' + scale]
            decoder['features_v_gt_dx_' + scale] = tf.pad(
                decoder['input_v_' + scale],
                [[0, 0], [0, 0], [0, 0], [0, 1], [0, 0]])[:, :, :,
                                                   1:, :] - decoder[
                                                       'input_v_' + scale]
            decoder['features_v_gt_dy_' + scale] = tf.pad(
                decoder['input_v_' + scale],
                [[0, 0], [0, 0], [0, 1], [0, 0], [0, 0]])[:, :, 1:,
                                                   :, :] - decoder[
                                                       'input_v_' + scale]
            decoder['features_h_gt_dd_' + scale] = tf.pad(
                decoder['input_h_' + scale],
                [[0, 0], [0, 1], [0, 0], [0, 0], [0, 0]])[:, 1:, :,
                                                   :, :] - decoder[
                                                       'input_h_' + scale]
            decoder['features_h_gt_dx_' + scale] = tf.pad(
                decoder['input_h_' + scale],
                [[0, 0], [0, 0], [0, 0], [0, 1], [0, 0]])[:, :, :,
                                                   1:, :] - decoder[
                                                       'input_h_' + scale]
            decoder['features_h_gt_dy_' + scale] = tf.pad(
                decoder['input_h_' + scale],
                [[0, 0], [0, 0], [0, 1], [0, 0], [0, 0]])[:, :, 1:,
                                                   :, :] - decoder[
                                                       'input_h_' + scale]

        self.decoders_3D[decoder_id] = decoder

  def create_discriminator(self):
      with tf.device('/device:GPU:%i' % (self.discriminator_config['gan_preferred_gpu'])):
          with tf.variable_scope('discriminator'):
              discriminator = dict()
              discriminator['variables'] = []

              layout = []
              # layout.insert(0, self.config.layer_config[0]['layout'][0])
              for i in range(0, len(self.config.layers['discriminator_3D'])):
                  layout.append(self.config.layers['encoder_3D'][i])

              layout[0]['conv'][-2] = self.discriminator_config['channels']*4
              # layout[0]['conv'][-2] = self.discriminator_config['channels']

              # create encoder variables
              last_layer = min(len(layout), self.max_layer)

              for i in range(0, last_layer):
                  layer_id = "discriminator_%i" % i
                  print('    creating 3D encoder variables ' + layer_id)
                  discriminator['variables'].append(layers.discriminator_variables(layer_id, layout[i]))

              for scale in self.scales:
                  no_relu = False
                  # discriminator['features_v_' + scale] = self.decoders_3D[self.id]['upconv_v_' + scale]
                  # discriminator['features_h_' + scale] = self.decoders_3D[self.id]['upconv_h_' + scale]
                  #
                  # discriminator['features_v_gt_' + scale] = self.decoders_3D[self.id]['input_v_' + scale]
                  # discriminator['features_h_gt_' + scale] = self.decoders_3D[self.id]['upconv_h_' + scale]

                  discriminator['features_v_' + scale] = tf.concat([ self.decoders_3D[self.id]['upconv_v_' + scale][:,0:-1,0:-1,0:-1,:],
                                                                     self.decoders_3D[self.id]['features_v_dd_' + scale][:,0:-1,0:-1,0:-1,:],
                                                                     self.decoders_3D[self.id]['features_v_dx_' + scale][:,0:-1,0:-1,0:-1,:],
                                                                     self.decoders_3D[self.id]['features_v_dy_' + scale][:,0:-1,0:-1,0:-1,:]], axis = -1)
                  discriminator['features_h_' + scale] = tf.concat([self.decoders_3D[self.id]['upconv_h_' + scale][:,0:-1,0:-1,0:-1,:],
                                                                    self.decoders_3D[self.id]['features_h_dd_' + scale][:,0:-1,0:-1,0:-1,:],
                                                                    self.decoders_3D[self.id]['features_h_dx_' + scale][:,0:-1,0:-1,0:-1,:],
                                                                    self.decoders_3D[self.id]['features_h_dy_' + scale][:,0:-1,0:-1,0:-1,:]], axis = -1)

                  discriminator['features_v_gt_' + scale] = tf.concat([self.decoders_3D[self.id]['input_v_' + scale][:,0:-1,0:-1,0:-1,:],
                                                                       self.decoders_3D[self.id]['features_v_gt_dd_' + scale][:,0:-1,0:-1,0:-1,:],
                                                                       self.decoders_3D[self.id]['features_v_gt_dx_' + scale][:,0:-1,0:-1,0:-1,:],
                                                                       self.decoders_3D[self.id]['features_v_gt_dy_' + scale][:,0:-1,0:-1,0:-1,:]], axis=-1)
                  discriminator['features_h_gt_' + scale] = tf.concat([self.decoders_3D[self.id]['input_h_' + scale][:,0:-1,0:-1,0:-1,:],
                                                                       self.decoders_3D[self.id]['features_h_gt_dd_' + scale][:,0:-1,0:-1,0:-1,:],
                                                                       self.decoders_3D[self.id]['features_h_gt_dx_' + scale][:,0:-1,0:-1,0:-1,:],
                                                                       self.decoders_3D[self.id]['features_h_gt_dy_' + scale][:,0:-1,0:-1,0:-1,:]], axis=-1)
                  if scale == 's2':
                    num_layers = last_layer-1
                  else:
                      num_layers = last_layer

                  for i in range(0, num_layers):
                      layer_id_v = "v_%s_%i" % ('discriminator_'+scale, i)
                      layer_id_h = "h_%s_%i" % ('discriminator_'+scale, i)
                      print('    generating downconvolution layer structure for %s %i' % ('discriminator', i))
                      discriminator['features_v_'+scale] = layers.discriminator_conv3d(layer_id_v,
                                                                                discriminator['variables'][i],
                                                                                discriminator['features_v_'+scale],
                                                                                self.phase,
                                                                                self.config.training,
                                                                                no_relu=no_relu).out
                      discriminator['features_h_'+scale] = layers.discriminator_conv3d(layer_id_h,
                                                                                discriminator['variables'][i],
                                                                                discriminator['features_h_'+scale],
                                                                                self.phase,
                                                                                self.config.training,
                                                                                no_relu=no_relu).out

                      discriminator['features_v_gt_'+scale] = layers.discriminator_conv3d(layer_id_v + '_gt',
                                                                                   discriminator['variables'][i],
                                                                                   discriminator['features_v_gt_'+scale],
                                                                                   self.phase,
                                                                                   self.config.training,
                                                                                   no_relu=no_relu).out
                      discriminator['features_h_gt_'+scale] = layers.discriminator_conv3d(layer_id_h + '_gt',
                                                                                   discriminator['variables'][i],
                                                                                   discriminator['features_h_gt_'+scale],
                                                                                   self.phase,
                                                                                   self.config.training,
                                                                                   no_relu=no_relu).out
                  # create dense layers
                  print('    creating dense layers for discriminator')
                  sh = discriminator['features_v_'+scale].shape.as_list()
                  discriminator['encoder_input_size_'+scale] = sh[1] * sh[2] * sh[3] * sh[4]
                  # setup shared feature space between horizontal/vertical encoder
                  discriminator['features_transposed_'+scale] = tf.concat(
                      [tf.reshape(tf.transpose(discriminator['features_h_'+scale], [0, 1, 3, 2, 4]),
                                  [-1, discriminator['encoder_input_size_'+scale]]),
                       tf.reshape(discriminator['features_v_'+scale], [-1, discriminator['encoder_input_size_'+scale]])], 1)

                  discriminator['features_transposed_gt_'+scale] = tf.concat(
                      [tf.reshape(tf.transpose(discriminator['features_h_gt_'+scale], [0, 1, 3, 2, 4]),
                                  [-1, discriminator['encoder_input_size_'+scale]]),
                       tf.reshape(discriminator['features_v_gt_'+scale], [-1, discriminator['encoder_input_size_'+scale]])], 1)

                  discriminator['discriminator_nodes_'+scale] = discriminator['features_transposed_'+scale].shape.as_list()[1]

                  discriminator['logits_sr_'+scale] = layers.bn_dense_discriminator(discriminator['features_transposed_'+scale],
                                                                             discriminator['discriminator_nodes_'+ scale],
                                                                             1, 'bn_gan_out_sr_'+scale)

                  discriminator['logits_gt_'+scale] = layers.bn_dense_discriminator(
                      discriminator['features_transposed_gt_'+scale], discriminator['discriminator_nodes_'+scale],
                      1, 'bn_gan_out_gt_'+scale)

              self.discriminator = discriminator

  def add_training_ops( self ):

    print( 'creating training ops' )

    # what needs to be updated before training
    self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # L2-loss on feature layers

    for cfg in self.config.minimizers:
        if 'losses_3D' in cfg:
            counter = 0
            for scale in self.scales:
                minimizer = dict()
                minimizer['id'] = cfg['id'] + '_' + scale
                print('  minimizer ' + minimizer['id'])

                with tf.device('/device:GPU:%i' % (cfg['preferred_gpu'][counter])):
                    counter +=1
                    minimizer['loss_' + scale] = 0
                    minimizer['requires'] = []
                    minimizer['requires'].append(scale)
                    for id in cfg['losses_3D']:
                        if self.decoders_3D[id]['train']:
                            minimizer['loss_'+scale] += self.decoders_3D[id]['weight'] * (self.decoders_3D[id]['loss_'+scale] +self.decoders_3D[id]['diffloss_'+scale])+\
                                           self.decoders_3D[id]['adv_loss_weight']*self.decoders_3D[id]['loss_adv_'+scale] + \
                                           self.decoders_3D[id]['percep_loss_weight'] * self.decoders_3D[id]['loss_p_' + scale]

                    with tf.control_dependencies( self.update_ops ):
                        # Ensures that we execute the update_ops before performing the train_step
                        minimizer['orig_optimizer'] = tf.train.AdamOptimizer(cfg['step_size'])
                        minimizer['optimizer'] = tf.contrib.estimator.clip_gradients_by_norm(minimizer['orig_optimizer'],
                                                                                           clip_norm=100.0)
                        minimizer['train_step'] = minimizer['optimizer'].minimize(minimizer['loss_'+scale],
                                                                                  var_list=[v for v in tf.global_variables() if "discriminator" not in v.name])

                    self.minimizers[ cfg[ 'id' ]+'_'+scale ] = minimizer

    if self.use_gan:
        minimizer = dict()
        minimizer['loss'] = 0
        minimizer['requires'] = []
        for scale in self.scales:
            minimizer['requires'].append(scale)
        minimizer['id'] = self.GAN_loss['id']
        print('  minimizer ' + minimizer['id'])
        with tf.device('/device:GPU:%i' % (self.discriminator_config[ 'gan_preferred_gpu' ])):
            minimizer['loss'] += self.discriminator_config['weight']*self.GAN_loss['loss']
        with tf.control_dependencies(self.update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            minimizer['orig_optimizer'] = tf.train.AdamOptimizer(self.discriminator_config['step_size'])
            minimizer['optimizer'] = tf.contrib.estimator.clip_gradients_by_norm(minimizer['orig_optimizer'],
                                                                                 clip_norm=100.0)
            minimizer['train_step'] = minimizer['optimizer'].minimize(minimizer['loss'],
                                    var_list=[v for v in tf.global_variables() if "discriminator" in v.name])
        self.minimizers[minimizer['id']] = minimizer

  def resnet_forward(self, x, layer, scope):
    x = 255.0 * (0.5 * (x + 1.0))
    # subtract means
    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3],
                         name='img_mean')  # RGB means from VGG paper
    x = x - mean
    # send through resnet
    with slim.arg_scope(resnet_arg_scope()):
        _, layers = resnet_v1_50(x, num_classes=1000, is_training=False, reuse=self.reuse_resnet)
    self.reuse_resnet = True
    return layers['resnet_v1_50/' + layer]

  # def mobilenet_forward(self, x, layer, scope):
  #   x = tf.cast(x, tf.float32) * 2. - 1
  #   _, layers = mobilenet(x, num_classes=1001, depth_multiplier=1.0, is_training=False, reuse=self.reuse_mobilenet)
  #   self.reuse_mobilenet = True
  #   return layers[layer]

  # def mobilenet_forward(self, x, layer, scope):
  #   mean = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32, shape=[1, 1, 1, 3],
  #                        name='img_mean')
  #   x = tf.cast(x, tf.float32) * 2. - 1 +mean
  #   # send through resnet
  #   with slim.arg_scope(mobilenet_v1_arg_scope()):
  #       _, layers = mobilenet_v1(x, num_classes=1001, depth_multiplier=0.5, is_training=False, reuse=self.reuse_mobilenet)
  #   self.reuse_mobilenet = True
  #   return layers[layer]

  def inception_forward(self, x, layer, scope):
    mean = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32, shape=[1, 1, 1, 3],
                         name='img_mean')
    x = tf.cast(x, tf.float32) * 2. - 1 +mean
    # send through resnet
    with slim.arg_scope(inception_v3_arg_scope()):
        _, layers = inception_v3(x, num_classes=None, is_training=False, reuse=self.reuse_inception)
    self.reuse_inception = True
    return layers[layer]

  def vgg_forward(self, x, layer, scope):
    # apply vgg preprocessing
    # move to range 0-255
    x = 255.0 * (0.5 * (x + 1.0))
    # subtract means
    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean') # RGB means from VGG paper
    x = x - mean
    # convert to BGR
    x = x[:,:,:,::-1]
    # send through vgg19
    _,layers = vgg_19(x, is_training=False, reuse=self.reuse_vgg)
    self.reuse_vgg = True
    return layers['vgg_19/' + layer]

  # add training ops for additional decoder pathway (L2 loss)
  def setup_losses( self ):
   loss_summary = dict()

   for id in self.decoders_3D:
       # loss function for auto-encoder
       with tf.variable_scope('training_3D_' + id):
           if self.decoders_3D[id]['loss_fn'] == 'L2':
               counter = 0
               for scale in self.scales:
                   with tf.device('/device:GPU:%i' % (self.decoders_3D[id]['preferred_gpu'][counter])):
                       counter +=1
                       print('  creating L2-loss for refinement pipeline ' + id + ' ' + scale)
                       self.decoders_3D[id]['loss_'+scale] = 0
                       self.decoders_3D[id]['loss_v'] = tf.losses.mean_squared_error(self.decoders_3D[id]['input_v_'+scale],
                                                                                 self.decoders_3D[id][
                                                                                     'upconv_v_'+scale], weights=(1.0 + tf.exp(-tf.div(
                               self.decoders_3D[id]['input_v_'+scale],0.5))))
                       self.decoders_3D[id]['loss_h'] = tf.losses.mean_squared_error(self.decoders_3D[id]['input_h_'+scale],
                                                                                 self.decoders_3D[id][
                                                                                     'upconv_h_'+scale], weights=(1.0 + tf.exp(-tf.div(
                               self.decoders_3D[id]['input_h_'+scale],0.5))))
                       self.decoders_3D[id]['loss_'+scale] += self.decoders_3D[id]['loss_v'] + self.decoders_3D[id]['loss_h']
                       sh = self.decoders_3D[id]['input_h_'+scale].shape.as_list()
                       cv_mask = np.zeros([self.config.training['samples_per_batch'], self.D, sh[-3], sh[-2], sh[-1]])
                       cv_mask[:, 4, :, :, :] = 1
                       self.decoders_3D[id]['loss_cv'] = tf.losses.mean_squared_error(
                           self.decoders_3D[id]['upconv_v_'+scale],
                           tf.transpose(self.decoders_3D[id]['upconv_h_'+scale], perm=[0, 1, 3, 2, 4]),
                           weights=cv_mask)
                       self.decoders_3D[id]['loss_'+scale] += self.decoders_3D[id]['loss_cv']
                       loss_summary[id + '_' + scale] = tf.summary.scalar('loss_3D' + id+'_'+scale, self.decoders_3D[id]['loss_'+scale])

                       # diffloss
                       self.decoders_3D[id]['diffloss_' + scale] = 0
                       self.decoders_3D[id]['diffloss_v'] = tf.losses.mean_squared_error(tf.concat(
                           [self.decoders_3D[id]['features_v_dd_' + scale][:, 0:-1, 0:-1, 0:-1, :],
                            self.decoders_3D[id]['features_v_dx_' + scale][:, 0:-1, 0:-1, 0:-1, :],
                            self.decoders_3D[id]['features_v_dy_' + scale][:, 0:-1, 0:-1, 0:-1, :]], axis=-1),
                           tf.concat(
                               [self.decoders_3D[id]['features_v_gt_dd_' + scale][:, 0:-1, 0:-1, 0:-1, :],
                                self.decoders_3D[id]['features_v_gt_dx_' + scale][:, 0:-1, 0:-1, 0:-1, :],
                                self.decoders_3D[id]['features_v_gt_dy_' + scale][:, 0:-1, 0:-1, 0:-1, :]],
                               axis=-1))
                       self.decoders_3D[id]['diffloss_h'] = tf.losses.mean_squared_error(tf.concat(
                           [self.decoders_3D[id]['features_h_dd_' + scale][:, 0:-1, 0:-1, 0:-1, :],
                            self.decoders_3D[id]['features_h_dx_' + scale][:, 0:-1, 0:-1, 0:-1, :],
                            self.decoders_3D[id]['features_h_dy_' + scale][:, 0:-1, 0:-1, 0:-1, :]], axis=-1),
                           tf.concat(
                               [self.decoders_3D[id]['features_h_gt_dd_' + scale][:, 0:-1, 0:-1, 0:-1, :],
                                self.decoders_3D[id]['features_h_gt_dx_' + scale][:, 0:-1, 0:-1, 0:-1, :],
                                self.decoders_3D[id]['features_h_gt_dy_' + scale][:, 0:-1, 0:-1, 0:-1, :]], axis=-1))
                       self.decoders_3D[id]['diffloss_' + scale] += self.decoders_3D[id]['diffloss_v'] + \
                                                                    self.decoders_3D[id][
                                                                        'diffloss_h']

                       loss_summary[id + '_diff_' + scale] = tf.summary.scalar('diffloss_3D' + id + '_' + scale,
                                                                          self.decoders_3D[id]['diffloss_' + scale])

           if self.decoders_3D[id]['loss_fn'] == 'L1':
               counter = 0
               for scale in self.scales:
                   with tf.device('/device:GPU:%i' % (self.decoders_3D[id]['preferred_gpu'][counter])):
                       counter += 1
                       print('  creating L1-loss for refinement pipeline ' + id + ' ' + scale)
                       self.decoders_3D[id]['loss_'+scale] = 0
                       self.decoders_3D[id]['loss_v'] = tf.losses.absolute_difference(self.decoders_3D[id]['input_v_'+scale],
                                                                                 self.decoders_3D[id][
                                                                                     'upconv_v_'+scale])
                       self.decoders_3D[id]['loss_h'] = tf.losses.absolute_difference(self.decoders_3D[id]['input_h_'+scale],
                                                                                 self.decoders_3D[id][
                                                                                     'upconv_h_'+scale])
                       self.decoders_3D[id]['loss_'+scale] += self.decoders_3D[id]['loss_v'] + self.decoders_3D[id]['loss_h']
                       sh = self.decoders_3D[id]['input_h_'+scale].shape.as_list()
                       cv_mask = np.zeros([self.config.training['samples_per_batch'], self.D, sh[-3], sh[-2], sh[-1]])
                       cv_mask[:, 4, :, :, :] = 1
                       self.decoders_3D[id]['loss_cv'] = tf.losses.absolute_difference(
                           self.decoders_3D[id]['upconv_v_'+scale],
                           tf.transpose(self.decoders_3D[id]['upconv_h_'+scale], perm=[0, 1, 3, 2, 4]),
                           weights=cv_mask)
                       self.decoders_3D[id]['loss_'+scale] += self.decoders_3D[id]['loss_cv']
                       loss_summary[id + '_' + scale] = tf.summary.scalar('loss_3D' + id+'_'+scale, self.decoders_3D[id]['loss_'+scale])

           for scale in self.scales:
               self.decoders_3D[id]['loss_adv_'+scale] = 0

           if self.use_gan:
               counter = 0
               for scale in self.scales:
                   with tf.device('/device:GPU:%i' % (self.decoders_3D[id]['preferred_gpu'][counter])):
                       counter += 1
                       self.decoders_3D[id]['loss_adv_'+scale] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                           logits=self.discriminator['logits_sr_'+scale],
                           labels=tf.ones_like(self.discriminator['logits_sr_'+scale])))
                       loss_summary['GAN_ADV_'+scale] =tf.summary.scalar('loss_adv' + id + '_'+scale, self.decoders_3D[id]['loss_adv_'+scale])

       for scale in self.scales:
           self.decoders_3D[id]['loss_p_' + scale] = 0

       if len(self.decoders_3D[id]['percep_loss']) > 0:
           if self.decoders_3D[id]['loss_fn'] == 'L1':
               for scale in self.scales:
                   for p_layer in self.decoders_3D[id]['percep_loss']:
                       # with tf.name_scope('inception_v3') as scope:
                       #     inception_y_v = tf.stack([resize_bicubic(self.inception_forward(
                       #         self.decoders_3D[id]['input_v_' + scale][:, view, ...], p_layer, scope), self.contect_features_size)
                       #                            for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('inception_v3') as scope:
                       #     inception_y_pred_v = tf.stack([resize_bicubic(self.inception_forward(
                       #         self.decoders_3D[id]['upconv_v_' + scale][:, view, ...], p_layer, scope), self.contect_features_size)
                       #                                 for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('inception_v3') as scope:
                       #     inception_y_h = tf.stack([resize_bicubic(self.inception_forward(
                       #         self.decoders_3D[id]['input_h_' + scale][:, view, ...], p_layer, scope), self.contect_features_size)
                       #         for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('inception_v3') as scope:
                       #     inception_y_pred_h = tf.stack([resize_bicubic(self.inception_forward(
                       #         self.decoders_3D[id]['upconv_h_' + scale][:, view, ...], p_layer, scope), self.contect_features_size)
                       #         for view in range(0, self.D)], axis=1)
                       # with tf.variable_scope('training_2D_' + id):
                       #     self.decoders_3D[id]['loss_p_' + scale] += tf.losses.absolute_difference(inception_y_v,
                       #                                                                              inception_y_pred_v) + \
                       #                                                tf.losses.absolute_difference(inception_y_h,
                       #                                                                              inception_y_pred_h)
                       # with tf.name_scope('MobilenetV2') as scope:
                       #     mobilenet_y_v = tf.stack([self.mobilenet_forward(
                       #         self.decoders_3D[id]['input_v_' + scale][:, view, ...], p_layer, scope)
                       #                            for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('MobilenetV2') as scope:
                       #     mobilenet_y_pred_v = tf.stack([self.mobilenet_forward(
                       #         self.decoders_3D[id]['upconv_v_' + scale][:, view, ...], p_layer, scope)
                       #                                 for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('MobilenetV2') as scope:
                       #     mobilenet_y_h = tf.stack([self.mobilenet_forward(
                       #         self.decoders_3D[id]['input_h_' + scale][:, view, ...], p_layer, scope)
                       #         for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('MobilenetV2') as scope:
                       #     mobilenet_y_pred_h = tf.stack([self.mobilenet_forward(
                       #         self.decoders_3D[id]['upconv_h_' + scale][:, view, ...], p_layer, scope)
                       #         for view in range(0, self.D)], axis=1)
                       # with tf.variable_scope('training_2D_' + id):
                       #     self.decoders_3D[id]['loss_p_' + scale] += tf.losses.absolute_difference(mobilenet_y_v,
                       #                                                                              mobilenet_y_pred_v) + \
                       #                                                tf.losses.absolute_difference(mobilenet_y_h,
                       #                                                                              mobilenet_y_pred_h)
                       # with tf.name_scope('vgg_19') as scope:
                       #     resnet_y_v = tf.stack([self.vgg_forward(
                       #         self.decoders_3D[id]['input_v_' + scale][:, view, ...], p_layer, scope)
                       #                            for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('vgg_19') as scope:
                       #     resnet_y_pred_v = tf.stack([self.vgg_forward(
                       #         self.decoders_3D[id]['upconv_v_' + scale][:, view, ...], p_layer, scope)
                       #                                 for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('vgg_19') as scope:
                       #     resnet_y_h = tf.stack([self.vgg_forward(
                       #         self.decoders_3D[id]['input_h_' + scale][:, view, ...], p_layer, scope)
                       #         for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('vgg_19') as scope:
                       #     resnet_y_pred_h = tf.stack([self.vgg_forward(
                       #         self.decoders_3D[id]['upconv_h_' + scale][:, view, ...], p_layer, scope)
                       #         for view in range(0, self.D)], axis=1)
                       # with tf.variable_scope('training_2D_' + id):
                       #     self.decoders_3D[id]['loss_p_' + scale] += tf.losses.absolute_difference(resnet_y_v,
                       #                                                                              resnet_y_pred_v) + \
                       #                                                tf.losses.absolute_difference(resnet_y_h,
                       #                                                                              resnet_y_pred_h)
                       with tf.name_scope('resnet_v1_50') as scope:
                           resnet_y_v = tf.stack([self.resnet_forward(self.decoders_3D[id]['input_v_' + scale][:,view,...], p_layer, scope)
                                                for view in range(0,self.D)], axis = 1)
                       with tf.name_scope('resnet_v1_50') as scope:
                           resnet_y_pred_v = tf.stack([self.resnet_forward(self.decoders_3D[id]['upconv_v_' + scale][:,view,...], p_layer, scope)
                                                     for view in range(0, self.D)], axis=1)
                       with tf.name_scope('resnet_v1_50') as scope:
                           resnet_y_h = tf.stack([self.resnet_forward(
                               self.decoders_3D[id]['input_h_' + scale][:, view, ...], p_layer, scope)
                                                  for view in range(0, self.D)], axis=1)
                       with tf.name_scope('resnet_v1_50') as scope:
                           resnet_y_pred_h = tf.stack([self.resnet_forward(
                               self.decoders_3D[id]['upconv_h_' + scale][:, view, ...], p_layer, scope)
                                                       for view in range(0, self.D)], axis=1)
                       with tf.variable_scope('training_2D_' + id):
                           self.decoders_3D[id]['loss_p_' + scale] += tf.losses.absolute_difference(resnet_y_v,
                                                                                                    resnet_y_pred_v) + \
                                                                      tf.losses.absolute_difference(resnet_y_h,
                                                                                                    resnet_y_pred_h)
                   loss_summary['loss_p_' + scale] = tf.summary.scalar('loss_p_' + id + '_' + scale,
                                                                               self.decoders_3D[id][
                                                                                   'loss_p_' + scale])
           if self.decoders_3D[id]['loss_fn'] == 'L2':
               for scale in self.scales:
                   for p_layer in self.decoders_3D[id]['percep_loss']:
                       # with tf.name_scope('inception_v3') as scope:
                       #     inception_y_v = tf.stack([tf.reduce_mean(self.inception_forward(
                       #         self.decoders_3D[id]['input_v_' + scale][:, view, ...], p_layer, scope), axis = [1,2])
                       #                            for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('inception_v3') as scope:
                       #     inception_y_pred_v = tf.stack([tf.reduce_mean(self.inception_forward(
                       #         self.decoders_3D[id]['upconv_v_' + scale][:, view, ...], p_layer, scope), axis = [1,2])
                       #                                 for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('inception_v3') as scope:
                       #     inception_y_h = tf.stack([tf.reduce_mean(self.inception_forward(
                       #         self.decoders_3D[id]['input_h_' + scale][:, view, ...], p_layer, scope), axis = [1,2])
                       #         for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('inception_v3') as scope:
                       #     inception_y_pred_h = tf.stack([tf.reduce_mean(self.inception_forward(
                       #         self.decoders_3D[id]['upconv_h_' + scale][:, view, ...], p_layer, scope), axis = [1,2])
                       #         for view in range(0, self.D)], axis=1)
                       # with tf.variable_scope('training_2D_' + id):
                       #     self.decoders_3D[id]['loss_p_' + scale] += tf.losses.mean_squared_error(inception_y_v,
                       #                                                                              inception_y_pred_v) + \
                       #                                                tf.losses.mean_squared_error(inception_y_h,
                       #                                                                              inception_y_pred_h)
                       # with tf.name_scope('MobilenetV2') as scope:
                       #     mobilenet_y_v = tf.stack([self.mobilenet_forward(
                       #         self.decoders_3D[id]['input_v_' + scale][:, view, ...], p_layer, scope)
                       #                            for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('MobilenetV2') as scope:
                       #     mobilenet_y_pred_v = tf.stack([self.mobilenet_forward(
                       #         self.decoders_3D[id]['upconv_v_' + scale][:, view, ...], p_layer, scope)
                       #                                 for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('MobilenetV2') as scope:
                       #     mobilenet_y_h = tf.stack([self.mobilenet_forward(
                       #         self.decoders_3D[id]['input_h_' + scale][:, view, ...], p_layer, scope)
                       #         for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('MobilenetV2') as scope:
                       #     mobilenet_y_pred_h = tf.stack([self.mobilenet_forward(
                       #         self.decoders_3D[id]['upconv_h_' + scale][:, view, ...], p_layer, scope)
                       #         for view in range(0, self.D)], axis=1)
                       # with tf.variable_scope('training_2D_' + id):
                       #     self.decoders_3D[id]['loss_p_' + scale] += tf.losses.mean_squared_error(mobilenet_y_v,
                       #                                                                              mobilenet_y_pred_v) + \
                       #                                                tf.losses.mean_squared_error(mobilenet_y_h,
                       #                                                                              mobilenet_y_pred_h)
                       # with tf.name_scope('vgg_19') as scope:
                       #     resnet_y_v = tf.stack([self.vgg_forward(
                       #         self.decoders_3D[id]['input_v_' + scale][:, view, ...], p_layer, scope)
                       #                            for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('vgg_19') as scope:
                       #     resnet_y_pred_v = tf.stack([self.vgg_forward(
                       #         self.decoders_3D[id]['upconv_v_' + scale][:, view, ...], p_layer, scope)
                       #                                 for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('vgg_19') as scope:
                       #     resnet_y_h = tf.stack([self.vgg_forward(
                       #         self.decoders_3D[id]['input_h_' + scale][:, view, ...], p_layer, scope)
                       #         for view in range(0, self.D)], axis=1)
                       # with tf.name_scope('vgg_19') as scope:
                       #     resnet_y_pred_h = tf.stack([self.vgg_forward(
                       #         self.decoders_3D[id]['upconv_h_' + scale][:, view, ...], p_layer, scope)
                       #         for view in range(0, self.D)], axis=1)
                       # with tf.variable_scope('training_2D_' + id):
                       #     self.decoders_3D[id]['loss_p_' + scale] += tf.losses.mean_squared_error(resnet_y_v,
                       #                                                                              resnet_y_pred_v) + \
                       #                                                tf.losses.mean_squared_error(resnet_y_h,
                       #                                                                              resnet_y_pred_h)
                       with tf.name_scope('resnet_v1_50') as scope:
                           resnet_y_v = tf.stack([tf.reduce_mean(self.resnet_forward(
                               self.decoders_3D[id]['input_v_' + scale][:, view, ...], p_layer, scope), axis = [1,2])
                                                  for view in range(0, self.D)], axis=1)
                       with tf.name_scope('resnet_v1_50') as scope:
                           resnet_y_pred_v = tf.stack([tf.reduce_mean(self.resnet_forward(
                               self.decoders_3D[id]['upconv_v_' + scale][:, view, ...], p_layer, scope), axis = [1,2])
                                                       for view in range(0, self.D)], axis=1)
                       with tf.name_scope('resnet_v1_50') as scope:
                           resnet_y_h = tf.stack([tf.reduce_mean(self.resnet_forward(
                               self.decoders_3D[id]['input_h_' + scale][:, view, ...], p_layer, scope), axis = [1,2])
                               for view in range(0, self.D)], axis=1)
                       with tf.name_scope('resnet_v1_50') as scope:
                           resnet_y_pred_h = tf.stack([tf.reduce_mean(self.resnet_forward(
                               self.decoders_3D[id]['upconv_h_' + scale][:, view, ...], p_layer, scope), axis = [1,2])
                               for view in range(0, self.D)], axis=1)
                       with tf.variable_scope('training_2D_' + id):
                           self.decoders_3D[id]['loss_p_' + scale] += tf.losses.mean_squared_error(resnet_y_v,
                                                                                                    resnet_y_pred_v) + \
                                                                      tf.losses.mean_squared_error(resnet_y_h,
                                                                                                    resnet_y_pred_h)
                   loss_summary['loss_p_' + scale] = tf.summary.scalar('loss_p_' + id + '_' + scale,
                                                                          self.decoders_3D[id]['loss_p_' + scale])

   if self.use_gan:
       with tf.device( '/device:GPU:%i' % ( self.discriminator_config[ 'gan_preferred_gpu' ] )):
           with tf.variable_scope('discriminator'):
               GAN_loss = dict()
               GAN_loss['id'] = 'GAN'
               GAN_loss['loss'] = 0
               for scale in self.scales:
                   loss_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(self.discriminator['logits_gt_'+scale]),
                                                                              self.discriminator['logits_gt_'+scale]))
                   loss_fake = tf.reduce_mean(
                       tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.discriminator['logits_sr_'+scale]),
                                                       self.discriminator['logits_sr_'+scale]))
                   GAN_loss['loss'] += (loss_real + loss_fake)/2

               self.GAN_loss = GAN_loss

               loss_summary['GAN'] = tf.summary.scalar('loss_GAN', self.GAN_loss['loss'])
   image_summary = dict()
   if self.config.config['ColorSpace'] == 'YCBCR':
       for scale in self.scales:
           image_summary['lf_res_v_' + scale] =tf.summary.image('lf_res_v_' + scale, tf.clip_by_value(self.decoders_3D['Y']['upconv_v_' + scale][:, 4, :, :, 0:1],0.0,1.0),
                            max_outputs=3)
           image_summary['lf_res_h_' + scale] =tf.summary.image('lf_res_h_' + scale, tf.clip_by_value(self.decoders_3D['Y']['upconv_h_' + scale][:, 4, :, :, 0:1],0.0,1.0), max_outputs = 3)
           image_summary['lf_input_v_' + scale] = tf.summary.image('lf_input_v_' + scale,
                                                                   tf.clip_by_value(self.decoders_3D['Y']['input_v_' + scale][
                                                                   :,
                                                                   4, :, :, 0:1],0.0,1.0),
                                                                   max_outputs=3)
           image_summary['lf_input_h_' + scale] = tf.summary.image('lf_input_h_' + scale,
                                                                   tf.clip_by_value(self.decoders_3D['Y']['input_h_' + scale][
                                                                   :,
                                                                   4, :, :, 0:1],0.0,1.0), max_outputs=3)
       image_summary['SR_input'] =tf.summary.image('SR_input', self.stack_v[:, 2, :, :, 0:3], max_outputs=3)

   if self.config.config['ColorSpace'] == 'RGB':

       for scale in self.scales:
           image_summary['lf_res_v_' + scale] = tf.summary.image('lf_res_v_' + scale, tf.clip_by_value(
               self.decoders_3D[self.config.config['ColorSpace']]['upconv_v_' + scale][:, 4, :, :, :], 0.0, 1.0),
                                                                 max_outputs=3)
           image_summary['lf_res_h_' + scale] = tf.summary.image('lf_res_h_' + scale, tf.clip_by_value(
               self.decoders_3D[self.config.config['ColorSpace']]['upconv_h_' + scale][:, 4, :, :, :], 0.0, 1.0), max_outputs=3)

           image_summary['lf_input_v_' + scale] = tf.summary.image('lf_input_v_' + scale,
                                                                   self.decoders_3D[self.config.config['ColorSpace']]['input_v_' + scale][
                                                                   :,
                                                                   4, :, :, :],
                                                                   max_outputs=3)
           image_summary['lf_input_h_' + scale] = tf.summary.image('lf_input_h_' + scale,
                                                                   self.decoders_3D[self.config.config['ColorSpace']]['input_h_' + scale][
                                                                   :,
                                                                   4, :, :, :], max_outputs=3)

       image_summary['SR_input'] = tf.summary.image('SR_input', self.stack_v[:, 2, :, :, 0:3], max_outputs=3)

   self.merged_images = tf.summary.merge([ v for k,v in image_summary.items()])
   self.merged_s2 = tf.summary.merge(
           [v for k, v in loss_summary.items() if (k.endswith('s2') )])
   if 's4' in self.scales:
       self.merged_s4 = tf.summary.merge(
           [v for k, v in loss_summary.items() if (k.endswith('s4') )])
   if self.use_gan:
       self.merged_gan = tf.summary.merge([ v for k,v in loss_summary.items() if k.startswith('GAN')])

  # initialize new variables
  def initialize_uninitialized( self, sess ):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    
    for i in not_initialized_vars:
      print( str(i.name) )

    if len(not_initialized_vars):
      sess.run(tf.variables_initializer(not_initialized_vars))

  # prepare input
  def prepare_net_input( self, batch ):
      net_in = {  self.keep_prob:       1.0,
                  self.phase:           False,
                  self.noise_sigma:     self.config.training[ 'noise_sigma' ] }

      # bind 2D decoder inputs to batch stream
      sh = batch['stacks_v'].shape

      for id in self.decoders_3D:
          decoder = self.decoders_3D[id]
          net_in[decoder['stack_h']] = batch['stacks_bicubic_h'][...,0:self.decoders_3D[self.id]['channels']]
          net_in[decoder['stack_v']] = batch['stacks_bicubic_v'][...,0:self.decoders_3D[self.id]['channels']]
          for scale in self.scales:
              if 'input_v_'+scale in decoder:
                idx_v = 'stacks_v_'+scale
                idx_h = 'stacks_h_' + scale
                if idx_v in batch:
                    net_in[decoder['input_v_'+scale]] = batch[idx_v][:,:,:,:,0:self.decoders_3D[self.id]['channels']]
                    net_in[decoder['input_h_' + scale]] = batch[idx_h][:,:,:,:,0:self.decoders_3D[self.id]['channels']]
                else:
                    if scale == 's2':
                        net_in[decoder['input_v_'+scale]] = np.zeros((sh[0], self.D, self.H_s2,self.W_s2,self.decoders_3D[self.id]['channels']), np.float32)
                        net_in[decoder['input_h_' + scale]] = np.zeros((sh[0], self.D, self.H_s2, self.W_s2, self.decoders_3D[self.id]['channels']),
                                                                          np.float32)
                    if scale == 's4':
                        net_in[decoder['input_v_' + scale]] = np.zeros((sh[0], self.D,self.H_s4, self.W_s4, self.decoders_3D[self.id]['channels']), np.float32)
                        net_in[decoder['input_h_' + scale]] = np.zeros((sh[0], self.D, self.H_s4, self.W_s4, self.decoders_3D[self.id]['channels']),
                                                                          np.float32)

      net_in[self.stack_v] = batch['stacks_v'][:,0:9:4,:,:,0:1]
      net_in[self.stack_h] = batch['stacks_h'][:,0:9:4,:,:,0:1]

      return net_in
