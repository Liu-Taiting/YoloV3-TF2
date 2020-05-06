'''
@description: 
@author: liutaiting
@lastEditors: liutaiting
@Date: 2020-05-05 10:55:07
@LastEditTime: 2020-05-05 11:09:00
'''

from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.YOLO = edict()

# Set the class name
__C.YOLO.CLASSES              = "./data/classes/yymnist.names"
__C.YOLO.ANCHORS              = "./data/anchors/basline_anchors.txt"
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5