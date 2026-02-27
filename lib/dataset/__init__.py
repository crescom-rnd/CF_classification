# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
from .coco import COCODataset as coco
# from .coco import COCO as coco

from .cervical_1st import CERVICAL_1ST as cervical_1st
from .compress_fracture import COMPRESS_FRACTURE as cf