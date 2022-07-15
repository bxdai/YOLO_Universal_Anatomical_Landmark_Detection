from __future__ import absolute_import#优先查找系统中的路径
from . import *

from model.datasets.cephalometric import Cephalometric
from model.datasets.hand import Hand
from model.datasets.chest import Chest

def get_dataset(s):
    return {
            'cephalometric':Cephalometric,
            'hand':Hand,
            'chest':Chest,
           }[s.lower()]


