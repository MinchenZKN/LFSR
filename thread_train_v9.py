#
# Thread for training the autoencoder network.
# Idea is that we can stream data in parallel, which is a bottleneck otherwise
#
import os
import datetime
import sys

import tensorflow as tf
import libs.tf_tools as tft

def trainer_thread( model_path, hp, inputs ):

  # start tensorflow session
  session_config = tf.ConfigProto( allow_soft_placement=True,
                                   log_device_placement=hp.config[ 'log_device_placement' ] )
  sess = tf.InteractiveSession( config=session_config )

  # import network
  from   cnn_autoencoder_v9 import create_cnn
  cnn = create_cnn( hp )

  # add optimizers (will be saved with the network)
  cnn.add_training_ops()
  # start session
  print( '  initialising TF session' )
  sess.run(tf.global_variables_initializer())
  print('... restoring resnet50 ')
  # print('... restoring vgg19 ')
  # print('... restoring moobilenet ')
  # print('... restoring inception ')
  ckpt_resnet = 'resnet_v1_50.ckpt'
  # ckpt_vgg = 'vgg_19.ckpt'
  # ckpt_mobilenet = 'mobilenet/mobilenet_v1_0.5_192.ckpt'
  # ckpt_inception = 'inception/inception_v3.ckpt'

  # reader = tf.train.NewCheckpointReader(ckpt_mobilenet)
  # tst = reader.get_variable_to_shape_map()
  # reader.get_variable_to_shape_map()['MobilenetV2/expanded_conv_10/expand/weights']
  # import numpy as np
  # print(np.transpose(sorted(tst)))
  # resnet_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='resnet_v1_50'))
  # resnet_saver.restore(sess, ckpt_resnet)

  # vgg_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='vgg_19'))
  # vgg_saver.restore(sess, ckpt_vgg)

  # mobilenet_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='MobilenetV1'))
  # mobilenet_saver.restore(sess, ckpt_mobilenet)

  # inception_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='InceptionV3'))
  # inception_saver.restore(sess, ckpt_inception)

  # var_list = [op.name+':0' for op in sess.graph.get_operations() if
  #             'variable' in op.type.lower() and 'resnet' in op.name.lower()]
  # var_list = [sess.graph.get_tensor_by_name(t) for t in var_list]
  # saver = tf.train.Saver(var_list=var_list)
  # saver.restore(sess, ckpt_path)
  # test
  # (tst) = sess.run('resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights:0')
  print( '  ... done' )

  # save object
  print( '  checking for model ' + model_path )
  if os.path.exists( model_path + 'model.ckpt.index' ):
    print( '  restoring model ' + model_path )
    tft.optimistic_restore( sess,  model_path + 'model.ckpt' )
    print( '  ... done.' )
  else:
    print( '  ... not found.' )


  writerTensorboard = tf.summary.FileWriter(hp.tf_log_path + hp.network_model, sess.graph)
  # writerTensorboard = tf.summary.FileWriter('/home/mz/HD data/Tensorboard Logs/' + hp.network_model, sess.graph)
  # writerTensorboard = tf.summary.FileWriter('./visual_' + hp.network_model, sess.graph)
  # new saver object with complete network
  saver = tf.train.Saver()

  # statistics
  count = 0.0
  print( 'lf cnn trainer waiting for inputs' )


  terminated = 0
  counter = 0
  iterGan = hp.iterGan
  if len(hp.discriminator) > 0:
    use_gan = True
    train_gan = True
  else:
    use_gan = False
    train_gan = False

  while not terminated:

    batch = inputs.get()
    if batch == ():
            terminated = 1
    else:

      niter      = batch[ 'niter' ]
      ep         = batch[ 'epoch' ]

      # default params for network input
      net_in = cnn.prepare_net_input( batch )

      # evaluate current network performance on mini-batch
      # (tst_h,tst_v) = sess.run([cnn.refinement_3D['lf']['stack_h'],cnn.refinement_3D['lf']['stack_v']], feed_dict=net_in)
      if batch[ 'logging' ]:
        summary_image = sess.run(cnn.merged_images, feed_dict=net_in)
        writerTensorboard.add_summary(summary_image, niter)

        print()
        sys.stdout.write( '  dataset(%d:%s) ep(%d) batch(%d) : ' %(batch[ 'nfeed' ], batch[ 'feed_id' ], ep, niter) )

        #loss_average = (count * loss_average + loss) / (count + 1.0)
        count = count + 1.0
        fields=[ '%s' %( datetime.datetime.now() ), batch[ 'feed_id' ], batch[ 'nfeed' ], niter, ep ]

        for id in cnn.decoders_3D:
            ( loss ) = sess.run( cnn.decoders_3D[id]['loss_s2'], feed_dict=net_in )
            sys.stdout.write( '  %s %g   ' %(id, loss) )
            fields.append( id )
            fields.append( loss )
        for id in cnn.minimizers:
          ok = True
          for r in cnn.minimizers[id]['requires']:
            if not ('stacks_v_' + r in batch):
              ok = False
          if ok:
            if id.endswith('s2'):
              summary = sess.run(cnn.merged_s2, feed_dict=net_in)
            if id.endswith('s4'):
              summary = sess.run(cnn.merged_s4, feed_dict=net_in)
            if id=='GAN':
              summary = sess.run(cnn.merged_gan, feed_dict=net_in)
            writerTensorboard.add_summary(summary, niter)

        # (tst) = sess.run( cnn.tst, feed_dict=net_in )
        import csv
        with open( model_path + batch[ 'logfile' ], 'a+') as f:
          writer = csv.writer(f)
          writer.writerow(fields)

        print( '' )
        #code.interact( local=locals() )


      if batch[ 'niter' ] % hp.training[ 'save_interval' ] == 0 and niter != 0 and batch[ 'nfeed' ] == 0 and batch[ 'training' ]:
        # epochs now take too long, save every few 100 steps
        # Save the variables to disk.
        save_path = saver.save(sess, model_path + 'model_'+ str(batch[ 'niter' ])+'.ckpt' )
        print( 'NEXT EPOCH' )
        print("  model saved in file: %s" % save_path)
        # statistics
        #print("  past epoch average loss %g"%(loss_average))
        count = 0.0


      # run training step
      if batch[ 'training' ]:
        net_in[ cnn.phase ] = True
        #code.interact( local=locals() )
        sys.stdout.write( '.' ) #T%i ' % int(count) )
        if train_gan:
            ok = True
            for r in cnn.minimizers['GAN']['requires']:
                if not ('stacks_v_' + r in batch):
                    ok = False
            if ok:
              sys.stdout.write(cnn.minimizers['GAN']['id'] + ' ')
              sess.run(cnn.minimizers['GAN']['train_step'],
                       feed_dict=net_in)
            counter += 1
            if counter == iterGan:
              train_gan = False
        else:
          for id in cnn.minimizers:
            if id != 'GAN':
              ok = True
              for r in cnn.minimizers[id]['requires']:
                if not ('stacks_v_'+r in batch):
                  ok = False
              if ok:
                sys.stdout.write( cnn.minimizers[id][ 'id' ] + ' ' )
                sess.run( cnn.minimizers[id][ 'train_step' ],
                          feed_dict = net_in )
          if use_gan :
              train_gan = True
              counter = 0
        sys.stdout.flush()

    inputs.task_done()