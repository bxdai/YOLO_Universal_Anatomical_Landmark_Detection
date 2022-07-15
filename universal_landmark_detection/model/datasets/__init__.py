#from __future__ import absolute_import#有限查找系统中的路径
from . import *

from .cephalometric import Cephalometric
from .hand import Hand
from .chest import Chest

def get_dataset(s):
    return {
            'cephalometric':Cephalometric,
            'hand':Hand,
            'chest':Chest,
           }[s.lower()]


