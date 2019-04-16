import  numpy as  np
import tensorflow as tf
# tf.contrib.resampler

def refocus_meow_tf(cv, disp, nviews):
    height = cv.shape.as_list()[-3]
    width = cv.shape.as_list()[-2]
    c = (nviews + 1) / 2 - 1

    disp_v = tf.tile(tf.expand_dims(disp, axis = 1), [1,nviews,1,1,1])
    disp_h = tf.tile(tf.expand_dims(tf.transpose(disp, [0,2,1,3]), axis = 1), [1,nviews,1,1,1])
    cv_v = cv
    cv_h = tf.transpose(cv, [0, 2, 1, 3])

    [t, v, u] = tf.meshgrid(tf.range(nviews), tf.range(height), tf.range(width))
    tt = tf.transpose(t,[1, 0, 2])
    vv = tf.transpose(v,[1, 0, 2])
    uu = tf.transpose(u, [1, 0, 2])
    tt = tf.cast(tf.expand_dims(tt, axis=-1), np.float32)
    vv = tf.cast(tf.expand_dims(vv, axis=-1), np.float32)
    uu = tf.cast(tf.expand_dims(uu, axis=-1), np.float32)

    y_v = tf.map_fn(lambda x: tf.multiply(tt-c, x) + vv, disp_v)
    y_h = tf.map_fn(lambda x: tf.multiply(tt - c, x) + vv, disp_h)
    z = tf.map_fn(lambda x: tf.multiply(0.0, x) + uu, disp_v)

    y_v = tf.clip_by_value(y_v, 0, height - 1)
    y_h = tf.clip_by_value(y_h, 0, width - 1)

    warp_v= tf.concat([z,y_v], axis = -1)
    warp_h = tf.concat([z,y_h], axis=-1)

    stack_h = tf.contrib.resampler.resampler(cv_h, warp_h)
    stack_v = tf.contrib.resampler.resampler(cv_v, warp_v)

    return stack_h, stack_v

def refocus_custom_tf1(cv, disp, nviews):
    back_prop = False
    padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
    cv = tf.pad(cv, paddings=padding)
    disp = tf.pad(disp, paddings=padding)

    height = cv.shape.as_list()[-3]
    width = cv.shape.as_list()[-2]
    c = (nviews + 1) / 2 - 1

    disp_v = tf.tile(tf.expand_dims(disp, axis=1), [1, nviews, 1, 1, 1])
    disp_h = tf.tile(tf.expand_dims(tf.transpose(disp, [0, 2, 1, 3]), axis=1), [1, nviews, 1, 1, 1])

    cv_v = tf.tile(tf.expand_dims(cv, axis=1), [1, nviews, 1, 1, 1])
    cv_h = tf.tile(tf.expand_dims(tf.transpose(cv, [0, 2, 1, 3]), axis=1), [1, nviews, 1, 1, 1])

    [t, v, u] = tf.meshgrid(tf.range(nviews), tf.range(height), tf.range(width))
    tt = tf.transpose(t, [1, 0, 2])
    vv = tf.transpose(v, [1, 0, 2])
    uu = tf.transpose(u, [1, 0, 2])
    tt = tf.cast(tf.expand_dims(tt, axis=-1), np.float32)
    vv = tf.cast(tf.expand_dims(vv, axis=-1), np.float32)
    uu = tf.cast(tf.expand_dims(uu, axis=-1), np.float32)

    initializer = tf.zeros(uu.shape.as_list())

    y_v = tf.scan(lambda a,x: tf.multiply(tt - c, x) + vv, disp_v, initializer,back_prop=back_prop )
    y_h = tf.scan(lambda a,x: tf.multiply(tt - c, x) + vv, disp_h, initializer, back_prop=back_prop)
    x_x = tf.scan(lambda a,x: tf.multiply(0.0, x) + uu, disp_v, initializer, back_prop=back_prop)
    t_t = tf.scan(lambda a,x: tf.multiply(0.0, x) + tt, disp_v, initializer,back_prop=back_prop)

    y_v = tf.clip_by_value(y_v, 1, height - 2)
    y_h = tf.clip_by_value(y_h, 1, width - 2)

    # neighbours
    def get_neighbours(inputs, height):
        outputs0 = tf.floor(inputs)
        outputs1 = outputs0 + 1
        outputs0 = tf.clip_by_value(outputs0, 0, height - 1)
        outputs1 = tf.clip_by_value(outputs1, 0, height - 1)
        return outputs0, outputs1

    x0,x1 = get_neighbours(x_x, height)
    y_v0, y_v1 = get_neighbours(y_v, height)
    y_h0, y_h1 = get_neighbours(y_h, height)

    def get_interpolation(t_t, x0, x1, y0, y1, x, y, im):
        inds00 = tf.cast(tf.concat([t_t, y0, x0], axis=-1), np.int32)
        inds10 = tf.cast(tf.concat([t_t, y1, x0], axis=-1), np.int32)
        inds01 = tf.cast(tf.concat([t_t, y0, x1], axis=-1), np.int32)
        inds11 = tf.cast(tf.concat([t_t, y1, x1], axis=-1), np.int32)

        initializer = (tf.zeros(im.shape.as_list()[1:]),tf.cast(tf.zeros(inds00.shape.as_list()[1:]), np.int32))

        I00 = tf.scan(lambda a,x: (tf.gather_nd(x[0], x[1]), x[1]), (im, inds00), initializer,back_prop=back_prop)[0]
        I10 = tf.scan(lambda a,x: (tf.gather_nd(x[0], x[1]), x[1]), (im, inds10), initializer,back_prop=back_prop)[0]
        I01 = tf.scan(lambda a,x: (tf.gather_nd(x[0], x[1]), x[1]), (im, inds01), initializer,back_prop=back_prop)[0]
        I11 = tf.scan(lambda a,x: (tf.gather_nd(x[0], x[1]), x[1]), (im, inds11), initializer,back_prop=back_prop)[0]

        wa = tf.multiply(x1 - x,y1 - y)
        wb = tf.multiply(x1 - x,y - y0)
        wc = tf.multiply(x - x0,y1 - y)
        wd = tf.multiply(x - x0,y - y0)

        return tf.multiply(wa,I00) + tf.multiply(wb,I10) + tf.multiply(wc,I01) + tf.multiply(wd,I11)

    stack_v = get_interpolation(t_t, x0, x1, y_v0, y_v1, x_x, y_v, cv_v)
    stack_h = get_interpolation(t_t, x0, x1, y_h0, y_h1, x_x, y_h, cv_h)


    return stack_h[:,:,1:height - 1,1:width - 1,:], stack_v[:,:,1:height - 1,1:width - 1,:]

def bilinear_interp_3d(stack_h, stack_v):
    padding = [[0, 0], [0,1], [0, 0], [0, 0], [0, 0]]
    stack_h1 = tf.pad(stack_h, paddings=padding)
    stack_h1 = stack_h1[:,1:,:,:,:]
    stack_v1 = tf.pad(stack_v, paddings=padding)
    stack_v1 = stack_v1[:,1:,:,:,:]

    stack_h_mid = (stack_h + stack_h1) / 2
    stack_v_mid = (stack_v + stack_v1) / 2
    stack_h_first= (stack_h + stack_h_mid) / 2
    stack_v_first = (stack_v + stack_v_mid) / 2
    stack_h_third = (stack_h_mid + stack_h1) / 2
    stack_v_third = (stack_v_mid + stack_v1) / 2
    stack_h_out = tf.stack([stack_h[:,0,:,:,:],stack_h_first[:,0,:,:,:],
                            stack_h_mid[:,0,:,:,:],stack_h_third[:,0,:,:,:],
                            stack_h[:, 1, :, :, :], stack_h_first[:, 1, :, :, :],
                            stack_h_mid[:,1,:,:,:],stack_h_third[:,1,:,:,:],
                            stack_h[:, 2, :, :, :]], axis = 1)
    stack_v_out = tf.stack([stack_v[:,0,:,:,:],stack_v_first[:,0,:,:,:],
                            stack_v_mid[:,0,:,:,:],stack_v_third[:,0,:,:,:],
                            stack_v[:, 1, :, :, :], stack_v_first[:, 1, :, :, :],
                            stack_v_mid[:,1,:,:,:],stack_v_third[:,1,:,:,:],
                            stack_v[:, 2, :, :, :]], axis = 1)
    return(stack_h_out,stack_v_out)



def bilinear_interpolate_tf(im, t, x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    t = np.asarray(t)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[-1]-1)
    x1 = np.clip(x1, 0, im.shape[-1]-1)
    y0 = np.clip(y0, 0, im.shape[-2]-1)
    y1 = np.clip(y1, 0, im.shape[-2]-1)

    Ia = im[ t, y0, x0 ]
    Ib = im[ t, y1, x0 ]
    Ic = im[ t, y0, x1 ]
    Id = im[ t, y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id
