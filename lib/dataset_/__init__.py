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
from .spine import SPINEDataset as spine
from .scoliosis import SCOLIOSISDataset as scoliosis
from .scoliosis_2nd import SCOLIOSIS_2ndDataset as scoliosis_2nd
from .scoliosis_3rd import SCOLIOSIS_3rdDataset as scoliosis_3rd
from .scoliosis_4th import SCOLIOSIS_4thDataset as scoliosis_4th
from .scoliosis_2nd_all import SCOLIOSIS_2nd_all_Dataset as scoliosis_2nd_all
from .LLA_1st import LLA_1stDataset as LLA_1st
from .LLA_ankle import LLA_ankleDataset as LLA_ankle
from .LLA_fh import LLA_fhDataset as LLA_fh
from .LLA_knee import LLA_kneeDataset as LLA_knee
from .ba_distinguish import BA_distinguish as ba_distinguish
from .as2_1st import AS2_1ST as as2_1st
from .as2_2nd_cervical import AS2_2ND_CERVICAL as as2_2nd_cervical
from .as2_1st_lumbar import AS2_1ST_LUMBAR as as2_1st_lumbar
from .as2_2nd_lumbar import AS2_2ND_LUMBAR as as2_2nd_lumbar
from .lumbar_ap_1st import LUMBAR_AP_1ST as lumbar_ap_1st 
from .lumbar_ap_2nd import LUMBAR_AP_2ND as lumbar_ap_2nd
from .lumbar_ap_3rd import LUMBAR_AP_3RD as lumbar_ap_3rd
from .lumbar_lateral_1st import LUMBAR_LATERAL_1ST as lumbar_lateral_1st 
from .T import T as T
from .spine_background import SPINEBACKGROUNDDataset as spine_background
from .handbone import handboneDataset as handbone
from .fx import FX as fx
from .split_1st_2_lateral import SPLIT_1ST_2_LATERAL as split_1st_2_lateral
from .split_1st_1_lateral import SPLIT_1ST_1_LATERAL as split_1st_1_lateral



from .split_1st_2_AP import SPLIT_1ST_2_AP as split_1st_2_AP
from .split_1st_1_AP import SPLIT_1ST_1_AP as split_1st_1_AP

from .split_2nd_lateral import SPLIT_2ND_LATERAL as split_2nd_lateral
from .split_2nd_AP import SPLIT_2ND_AP as split_2nd_AP

from .ver2_lateral_1st import VER2_LATERAL_1ST as ver2_lateral_1st
from .ver2_lateral_2nd import VER2_LATERAL_2ND as ver2_lateral_2nd