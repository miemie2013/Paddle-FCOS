#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-09-05 18:21:37
#   Description : paddle_fcos
#
# ================================================================
from model.dla import *
from model.losses import *
from model.head import *
from model.neck import *
from model.resnet import *


def select_backbone(name):
    if name == 'Resnet':
        return Resnet
    if name == 'dla34':
        return dla34

def select_fpn(name):
    if name == 'FPN':
        return FPN

def select_head(name):
    if name == 'FCOSHead':
        return FCOSHead
    if name == 'FCOSSharedHead':
        return FCOSSharedHead

def select_loss(name):
    if name == 'FCOSLoss':
        return FCOSLoss




