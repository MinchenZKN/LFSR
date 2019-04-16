# define the configuration (hyperparameters) for the residual autoencoder
# for this type of network.


# NETWORK MODEL NAME
network_model = 'SR_LFDIFGAN_ycbcr_3d_1311_m1'

# data_path = '/data/cvia/Data_AWS/'
data_path = '/data/aa/Data_Cross_HR/'
# tf_log_path = '/home/mz/HD_data/Tensorboard Logs/'
# tf_log_path = '/data_fast/aa/tf_logs/meow/'
# tf_log_path = './tf_logs/'
tf_log_path = '/home/aa/tf_logs/meow/'
# tf_log_path = './tf_logs/'

# data_path = 'H:\\trainData\\'
# CURRENT TRAINING DATASET
training_data = [
    data_path + 'lf_patch_synthetic_rgb_sr_s4_1.hdf5',
    data_path + 'lf_patch_synthetic_rgb_sr_s4_1.hdf5',
    data_path + 'lf_patch_synthetic_rgb_sr_s4_STF_HCI.hdf5',
    data_path + 'lf_patch_synthetic_rgb_sr_s4_STF_HCI.hdf5',
    data_path + 'lf_patch_synthetic_rgb_sr_s4_3.hdf5',
    data_path + 'lf_patch_synthetic_rgb_sr_s4_3.hdf5',
    data_path + 'lf_patch_synthetic_rgb_sr_s4_flowers_2.hdf5',
    data_path + 'lf_patch_synthetic_rgb_sr_s4_flowers_2.hdf5',
    data_path + 'lf_patch_synthetic_rgb_sr_s4_2.hdf5',
    data_path + 'lf_patch_synthetic_rgb_sr_s4_2.hdf5',
    # '/home/aa/Python_projects/Data_train/super_resolution/lf_patch_synthetic_rgb_sr_s4_4.hdf5',
    # '/home/aa/Python_projects/Data_train/super_resolution/lf_patch_synthetic_rgb_sr_s4_4.hdf5',
    # data_path + 'lf_patch_synthetic_rgb_sr_s4_4.hdf5',
    # data_path + 'lf_patch_synthetic_rgb_sr_s4_4.hdf5',
    data_path + 'lf_patch_synthetic_rgb_sr_s4_disp_half_all.hdf5',
    data_path + 'lf_patch_synthetic_rgb_sr_s4_disp_half_all.hdf5',
]
# NETWORK LAYOUT HYPERPARAMETERS

# general config params
config = {
    # flag whether we want to train for RGB (might require more
    # changes in other files, can't remember right now)
    # 'ColorSpace'                  : 'YUV',
    'ColorSpace'                  : 'YCBCR',
    # 'ColorSpace'                  : 'LAB',
    # 'ColorSpace'                  : 'RGB',
    'VisibleGPU'                  :'0,1,2,3',
    # maximum layer which will be initialized (deprecated)
    'max_layer'            : 100,
    # add interpolated input as patch for skip connection in the upscaling phase
    'interpolate'          : False,
    # this will log every tensor being allocated,
    # very spammy, but useful for debugging
    'log_device_placement' : False,
}

# encoder for 48 x 48 patch, 9 views, RGB
D = 9
D_in = 3
H = 48
W = 48
# nviews = 9
H_s2 = 96
W_s2 = 96
H_s4 = 192
W_s4 = 192
cv_pos = int((D-1)/2)


# patch stepping
sx = 16
sy = 16
sx_s2 = 32
sy_s2 = 32
sx_HR = 64
sy_HR = 64

C = 3
C_value = 1
C_color = 2

# Number of features in the layers
# Number of features in the layers
L_half = 6
L = 8
L0 = 12
L1 = 16
L2 = 24
L3 = 32
L4 = 48
L5 = 64
L6 = 96
L7 = 128
L8 = 192
contect_features_size = [1,1]

# fraction between encoded patch and decoded patch. e.g.
# the feature maps of decoded patch are 3 times as many
# as the encoded patch, then patch_weight = 3

# Encoder stack for downwards convolution
patch_weight = 2

# chain of dense layers to form small bottleneck (can be empty)
layers = dict()
layer_config = [
    {
      'id': 'Y',# 'YUV', 'RGB', 'YCBCR', 'LAB' and any combinations
      'channels' : C,
      'start' : 0,
      'end': 3,
      'upscale': [
                {'conv': [3, 3, 3, L0, L0],
                 'stride': [1, 1, 1, 1, 1]
                 }],
      'final_s2': [
                {'conv': [1,1, 1, L * patch_weight, C],
                 'stride': [1,1, 1, 1, 1]
                 }, ],
      'final_s4': [
                {'conv': [1,1, 1, L_half * patch_weight, C],
                 'stride': [1,1, 1, 1, 1]
                 }, ],
    },
]



layers['encoder_3D'] = [
                {'conv': [3, 3, 3, C, L0],
                 'stride': [1, 1, 1, 1, 1]
                 },
                { 'conv'   : [ 3,3,3, L0, L1 ],
                  'stride' : [ 1,1, 2,2, 1 ]
                },
                # resolution now 9 x 18 x 18
                { 'conv'   : [ 3,3,3, L1, L2 ],
                  'stride' : [ 1,2, 1,1, 1 ]
                },
                # resolution now 5 x 18 x 18
                { 'conv'   : [ 3,3,3, L2, L3 ],
                  'stride' : [ 1,1, 2,2, 1 ]
                },
                # resolution now 5 x 9 x 9
                { 'conv'   : [ 3,3,3, L3, L3 ],
                  'stride' : [ 1,1, 1,1, 1 ]
                },
                # # resolution now 5 x 5 x 5
                { 'conv'   : [ 3,3,3, L3, L4 ],
                  'stride' : [ 1,1, 2,2, 1 ]
                },
                # resolution now 3 x 5 x 5
                {'conv'    : [3, 3, 3, L4, L5],
                 'stride'  : [1, 2, 1, 1, 1]
                },
                {'conv': [3, 3, 3, L5, L6],
                 'stride': [1, 1, 2, 2, 1]
                 },
                {'conv': [3, 3, 3, L6, L6],
                 'stride': [1, 1, 1, 1, 1]
                 },
                # resolution now 3 x 3 x 3
                ]

layers['discriminator_3D'] = [
                {'conv': [3, 3, 3, C, L],
                 'stride': [1, 1, 2, 2, 1]
                 },
                { 'conv'   : [ 3,3,3, L, L0 ],
                  'stride' : [ 1,1, 2,2, 1 ]
                },
                # resolution now 9 x 18 x 18
                { 'conv'   : [ 3,3,3, L0, L1 ],
                  'stride' : [ 1,1, 2,2, 1 ]
                },
                # resolution now 5 x 18 x 18
                { 'conv'   : [ 3,3,3, L1, L2 ],
                  'stride' : [ 1,2, 1,1, 1 ]
                },
                # # resolution now 5 x 5 x 5
                { 'conv'   : [ 3,3,3, L2, L3 ],
                  'stride' : [ 1,1, 2,2, 1 ]
                },
                # resolution now 3 x 5 x 5
                {'conv'    : [3, 3, 3, L3, L4],
                 'stride'  : [1, 2, 2, 2, 1]
                },
                {'conv': [3, 3, 3, L4, L4],
                 'stride': [1, 1, 2, 2, 1]
                 },
                # resolution now 3 x 3 x 3
                ]


layers['upscale'] = [
                { 'conv'   : [ 3, 3, 3, L, L0],
                  'stride' : [1, 1, 2, 2, 1],
                  'target_shape': [H_s2,W_s2],
                  'out': None,
                },
                { 'conv'   : [ 3, 3, 3, L, L],
                  'stride' : [1, 1, 1, 1, 1],
                  'target_shape': [H_s2, W_s2],
                  'out': None,
                },
                {'conv': [3, 3, 3, L, L],
                 'stride': [1, 1, 1, 1, 1],
                 'target_shape': [H_s2, W_s2],
                 'out': 's2',
                 },
                 {'conv': [3, 3, 3, L_half, L],
                  'stride': [1, 1, 2, 2, 1],
                  'target_shape': [H_s4, W_s4],
                 'out': None,
                 },
                 {'conv'    : [ 3, 3, 3, L_half, L_half],
                  'stride'  : [1, 1, 1, 1, 1],
                  'target_shape': [H_s4, W_s4],
                  'out': None,
                 },
                 {'conv': [3, 3, 3, L_half, L_half],
                  'stride': [1, 1, 1, 1, 1],
                  'target_shape': [H_s4, W_s4],
                  'out': 's4',
                  },
]


layers[ 'autoencoder_nodes' ] = []
layers[ '2D_encoder_nodes' ] = []
layers[ '2D_decoder_nodes' ] = []
layers[ 'preferred_gpu' ] = 0
layers[ 'merge_encoders' ] = False

discriminator = [
    {'gan_preferred_gpu':2,
     'step_size' : 1e-4,
     'train': True,
     'weight': 1.0,
     'channels': C,
     }
    ]
iterGan = 1

# 3D ENCODERS
encoders_3D = [
    {
      'id': 'Y',
      'channels': C,
      'preferred_gpu' : 0,
    },
]
#
# 2D DECODERS
#
# Each one generates a 2D upsampling pathway next to the
# two normal autoencoder pipes.
#
# Careful, takes memory. Remove some from training if limited.
#
decoders_3D = [
    {
        'id': 'Y',
        'channels': C,
        'preferred_gpu' : [0,2],
        'loss_fn':  'L2',
        'train':    True,
        'weight':   1.0,
        'no_relu': False,
        'skip_connection': True,
        'skip_id': ['Y'],
        'percep_loss': ['MaxPool_5a_3x3'],# 'block1','block2','block3','block4' 'conv4/conv4_4', 'conv5/conv5_4', 'conv2/conv2_2' ,'conv3/conv3_4','conv4/conv4_4', 'conv1/conv1_2',,'conv3/conv3_4','conv4/conv4_4','conv5/conv5_4' #['block1','block2', 'block3','block4'],
        'percep_loss_weight': 1.0,
        'adv_loss_weight': 1e-3,
    },
]

# MINIMIZERS
minimizers = [

    # center view super resolution
    {
        'id': 'Y_min',  # 'YCBCR_min'
        'losses_3D': ['Y'],  #  'KL_divergence' , 'YUV', 'RGB', 'YCBCR', 'LAB' and any combinations
        'optimizer': 'Adam',
        'preferred_gpu': [1,3],
        'step_size': 1e-4,
    },
 ]


# TRAINING HYPERPARAMETERS
training = dict()

# subsets to split training data into
# by default, only 'training' will be used for training, but the results
# on mini-batches on 'validation' will also be logged to check model performance.
# note, split will be performed based upon a random shuffle with filename hash
# as seed, thus, should be always the same for the same file.
#
training[ 'subsets' ] = {
  'validation'   : 0.05,
  'training'     : 0.95,
}


# number of samples per mini-batch
# reduce if memory is on the low side,
# but if it's too small, training becomes ridicuously slow
training[ 'samples_per_batch' ] = 6

# log interval (every # mini-batches per dataset)
training[ 'log_interval' ] = 5

# save interval (every # iterations over all datasets)
training[ 'save_interval' ] = 50

# noise to be added on each input patch
# (NOT on the decoding result)
training[ 'noise_sigma' ] = 0.0

# decay parameter for batch normalization
# should be larger for larger datasets
training[ 'batch_norm_decay' ]  = 0.9
# flag whether BN center param should be used
training[ 'batch_norm_center' ] = False
# flag whether BN scale param should be used
training[ 'batch_norm_scale' ]  = False
# flag whether BN should be zero debiased (extra param)
training[ 'batch_norm_zero_debias' ]  = False

eval_res = {
    'h_mask_s4': 150,
    'w_mask_s4': 150,
    'h_mask_s2': 70,
    'w_mask_s2': 70,
    'm_s4': 32,
    'm_s2': 16,
    'min_mask': 0.1,
    'result_folder': "./results/",
    'test_data_folder': "H:\\testData\\"
}
