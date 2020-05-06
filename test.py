'''
@description: 
@author: liutaiting
@lastEditors: liutaiting
@Date: 2020-05-05 11:03:39
@LastEditTime: 2020-05-05 11:07:44
'''

from core.config import cfg
from core.utils import read_calss_names

name = read_calss_names(cfg.YOLO.CLASSES)
print(name)