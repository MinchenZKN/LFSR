#
# Thread for evaluating the regression network.
# Reason for multithreading: Tensorflow can otherwise not coexist with
# other CUDA code.
#
import code


def evaluator_thread(cnn_file, hp, inputs, outputs, scale):
    # start tensorflow session
    import numpy as np
    import tensorflow as tf
    # sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=hp.config['log_device_placement'])
    sess = tf.InteractiveSession(config=session_config)

    # import network
    from cnn_autoencoder_v9 import create_cnn
    cnn = create_cnn(hp)

    # init session and load network state
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, cnn_file)

    # little chat
    print('lf cnn evaluator waiting for inputs')

    terminated = 0
    while not terminated:

        batch = inputs.get()
        if batch == ():
            terminated = 1
        else:
            out = dict()
            # default params for network input
            net_in = cnn.prepare_net_input(batch)
            net_in[cnn.noise_sigma] = 0.0

            # FOR SPECIFIC DECODER PATH (todo: make less of a hack)
            decoder_path = batch['decoder_path']
            decoder = cnn.decoders_3D[decoder_path]
            for i in range(0, len(scale)):
                (sv_v) = sess.run(decoder['upconv_v_' + scale[i]],
                                       feed_dict=net_in)
                (sv_h) = sess.run(decoder['upconv_h_' + scale[i]],
                                         feed_dict=net_in)
                #(sv_v) = sess.run(decoder['input_v_' + scale[i]],
                #                 feed_dict=net_in)
                #(sv_h) = sess.run(decoder['input_h_' + scale[i]],
                #                  feed_dict=net_in)
                out['SR_v_' + scale[i]] = sv_v
                out['SR_h_' + scale[i]] = np.transpose(sv_h, [0,1,3,2,4])

            outputs.put((out, batch))

        inputs.task_done()
