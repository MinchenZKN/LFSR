# define the configuration (hyperparameters) for the data format
# this is hard-coded now in all the training data, should not be changed anymore


import lf_tools
import numpy as np
import config_autoencoder_ycbcr_3d as hp

# general config params
# data_config = {
#     # patch size
#     'D' : 9,
#     'H' : 48,
#     'W' : 48,
#     'H_HR' : 96,
#     'W_HR' : 96,
#     # patch stepping
#     'SX' : 16,
#     'SY' : 16,
#     'SX_HR' : 32,
#     'SY_HR' : 32,
#     # depth range and number of labels
#     'dmin' : -3.5,
#     'dmax' :  3.5,
# }



# get patch at specified block coordinates
def get_patch( LF, by, bx, lf_scale ):

  patch = dict()

  # compute actual coordinates
  if lf_scale == 's2':
    y = by * hp.sy_s2
    x = bx * hp.sx_s2
    py = hp.H_s2
    px = hp.W_s2

  if lf_scale == 's4':
    y = by * hp.sy_HR
    x = bx * hp.sx_HR
    py = hp.H_s4
    px = hp.W_s4
  
  # extract data
  (stack_h, stack_v) = lf_tools.epi_stacks( LF, y, x, py, px )
  # make sure the direction of the view shift is the first spatial dimension
  stack_h = np.transpose( stack_h, (0, 2, 1, 3) )
  patch[ 'stack_v_HR' ] = stack_v
  patch[ 'stack_h_HR' ] = stack_h

  return patch
