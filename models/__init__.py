from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch

from .AttModel import *
from .basemodel import TopDownModel as TopDownModel_base

def setup(opt):
    if opt.caption_model == 'topdown':
        model = TopDownModel(opt)
    elif opt.caption_model == 'baseline':
        model = TopDownModel_base(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    return model
