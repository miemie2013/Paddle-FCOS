#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-09-05 18:21:37
#   Description : paddle_fcos。读取AdelaiDet FCOS_RT_MS_R_50_4x的权重。
#
# ================================================================
import numpy as np
from paddle import fluid
import torch

from config import *
from model.fcos import *
from model.head import *
from model.neck import *
from model.resnet import *




def load_weights(path):
    """ Loads weights from a compressed save file. """
    # state_dict = torch.load(path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    return state_dict

state_dict = load_weights('FCOS_RT_MS_R_50_4x_syncbn.pth')
print('============================================================')

backbone_dic = {}
fpn_dic = {}
fcos_head_dic = {}
others = {}
for key, value in state_dict.items():
    if 'tracked' in key:
        continue
    if 'bottom_up' in key:
        backbone_dic[key] = value.data.numpy()
    elif 'fpn' in key:
        fpn_dic[key] = value.data.numpy()
    elif 'fcos_head' in key:
        fcos_head_dic[key] = value.data.numpy()
    else:
        others[key] = value.data.numpy()

print()



cfg = FCOS_RT_R50_FPN_4x_Config()

# 创建模型
Backbone = select_backbone(cfg.backbone_type)
backbone = Backbone(**cfg.backbone)

Fpn = select_fpn(cfg.fpn_type)
fpn = Fpn(**cfg.fpn)

Loss = select_loss(cfg.fcos_loss_type)
fcos_loss = Loss(**cfg.fcos_loss)

Head = select_head(cfg.head_type)
head = Head(num_classes=80, fcos_loss=fcos_loss, nms_cfg=cfg.nms_cfg, **cfg.head)

fcos = FCOS(backbone, fpn, head)


image = L.data(name='image', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')
im_info = L.data(name='im_info', shape=[-1, 3], append_batch_size=False, dtype='float32')
outs = fcos(image, im_info)


# Create an executor using CPU as an example
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())



print('\nCopying...')


def copy_conv_bn(conv_name, w, scale, offset, m, v):
    tensor = fluid.global_scope().find_var('%s.conv.weight' % conv_name).get_tensor()
    tensor2 = fluid.global_scope().find_var('%s.bn.scale' % conv_name).get_tensor()
    tensor3 = fluid.global_scope().find_var('%s.bn.offset' % conv_name).get_tensor()
    tensor4 = fluid.global_scope().find_var('%s.bn.mean' % conv_name).get_tensor()
    tensor5 = fluid.global_scope().find_var('%s.bn.var' % conv_name).get_tensor()
    tensor.set(w, place)
    tensor2.set(scale, place)
    tensor3.set(offset, place)
    tensor4.set(m, place)
    tensor5.set(v, place)

def copy_conv_af(conv_name, w, scale, offset):
    tensor = fluid.global_scope().find_var('%s.conv.weight' % conv_name).get_tensor()
    tensor2 = fluid.global_scope().find_var('%s.af.scale' % conv_name).get_tensor()
    tensor3 = fluid.global_scope().find_var('%s.af.offset' % conv_name).get_tensor()
    tensor.set(w, place)
    tensor2.set(scale, place)
    tensor3.set(offset, place)

def copy_conv(conv_name, w, b):
    tensor = fluid.global_scope().find_var('%s.conv.weight' % conv_name).get_tensor()
    tensor2 = fluid.global_scope().find_var('%s.conv.bias' % conv_name).get_tensor()
    tensor.set(w, place)
    tensor2.set(b, place)

def copy_conv_gn(conv_name, w, b, scale, offset):
    tensor = fluid.global_scope().find_var('%s.conv.weight' % conv_name).get_tensor()
    tensor2 = fluid.global_scope().find_var('%s.conv.bias' % conv_name).get_tensor()
    tensor3 = fluid.global_scope().find_var('%s.gn.scale' % conv_name).get_tensor()
    tensor4 = fluid.global_scope().find_var('%s.gn.offset' % conv_name).get_tensor()
    tensor.set(w, place)
    tensor2.set(b, place)
    tensor3.set(scale, place)
    tensor4.set(offset, place)


# 获取FCOS模型的权重

resnet = backbone

# Resnet50
w = backbone_dic['backbone.bottom_up.stem.conv1.weight']
scale = backbone_dic['backbone.bottom_up.stem.conv1.norm.weight']
offset = backbone_dic['backbone.bottom_up.stem.conv1.norm.bias']
m = backbone_dic['backbone.bottom_up.stem.conv1.norm.running_mean']
v = backbone_dic['backbone.bottom_up.stem.conv1.norm.running_var']
copy_conv_bn('backbone.stage1.0.conv0', w, scale, offset, m, v)


nums = [3, 4, 6, 3]
for nid, num in enumerate(nums):
    stage_name = 'res' + str(nid + 2)
    for kk in range(num):
        conv_name1 = 'backbone.bottom_up.%s.%d.conv1' % (stage_name, kk)
        w = backbone_dic[conv_name1 + '.weight']
        scale = backbone_dic[conv_name1 + '.norm.weight']
        offset = backbone_dic[conv_name1 + '.norm.bias']
        m = backbone_dic[conv_name1 + '.norm.running_mean']
        v = backbone_dic[conv_name1 + '.norm.running_var']
        copy_conv_bn('backbone.stage%d.%d.conv0' % (2+nid, kk), w, scale, offset, m, v)

        conv_name2 = 'backbone.bottom_up.%s.%d.conv2' % (stage_name, kk)
        w = backbone_dic[conv_name2 + '.weight']
        scale = backbone_dic[conv_name2 + '.norm.weight']
        offset = backbone_dic[conv_name2 + '.norm.bias']
        m = backbone_dic[conv_name2 + '.norm.running_mean']
        v = backbone_dic[conv_name2 + '.norm.running_var']
        copy_conv_bn('backbone.stage%d.%d.conv1' % (2+nid, kk), w, scale, offset, m, v)

        conv_name3 = 'backbone.bottom_up.%s.%d.conv3' % (stage_name, kk)
        w = backbone_dic[conv_name3 + '.weight']
        scale = backbone_dic[conv_name3 + '.norm.weight']
        offset = backbone_dic[conv_name3 + '.norm.bias']
        m = backbone_dic[conv_name3 + '.norm.running_mean']
        v = backbone_dic[conv_name3 + '.norm.running_var']
        copy_conv_bn('backbone.stage%d.%d.conv2' % (2+nid, kk), w, scale, offset, m, v)

        # 每个stage的第一个卷积块才有4个卷积层
        if kk == 0:
            shortcut_name = 'backbone.bottom_up.%s.%d.shortcut' % (stage_name, kk)
            w = backbone_dic[shortcut_name + '.weight']
            scale = backbone_dic[shortcut_name + '.norm.weight']
            offset = backbone_dic[shortcut_name + '.norm.bias']
            m = backbone_dic[shortcut_name + '.norm.running_mean']
            v = backbone_dic[shortcut_name + '.norm.running_var']
            copy_conv_bn('backbone.stage%d.%d.conv3' % (2+nid, kk), w, scale, offset, m, v)
# fpn, 6个卷积层
w = fpn_dic['backbone.fpn_lateral5.weight']
b = fpn_dic['backbone.fpn_lateral5.bias']
copy_conv('fpn.s32_conv', w, b)

w = fpn_dic['backbone.fpn_lateral4.weight']
b = fpn_dic['backbone.fpn_lateral4.bias']
copy_conv('fpn.s16_conv', w, b)

w = fpn_dic['backbone.fpn_lateral3.weight']
b = fpn_dic['backbone.fpn_lateral3.bias']
copy_conv('fpn.s8_conv', w, b)

w = fpn_dic['backbone.fpn_output5.weight']
b = fpn_dic['backbone.fpn_output5.bias']
copy_conv('fpn.sc_s32_conv', w, b)

w = fpn_dic['backbone.fpn_output4.weight']
b = fpn_dic['backbone.fpn_output4.bias']
copy_conv('fpn.sc_s16_conv', w, b)

w = fpn_dic['backbone.fpn_output3.weight']
b = fpn_dic['backbone.fpn_output3.bias']
copy_conv('fpn.sc_s8_conv', w, b)


# head
num_convs = 4
ids = [[0, 1], [3, 4], [6, 7], [9, 10]]
for lvl in range(0, num_convs):
    # conv + gn
    w = fcos_head_dic['proposal_generator.fcos_head.cls_tower.%d.weight'%ids[lvl][0]]
    b = fcos_head_dic['proposal_generator.fcos_head.cls_tower.%d.bias'%ids[lvl][0]]
    scale = fcos_head_dic['proposal_generator.fcos_head.cls_tower.%d.weight'%ids[lvl][1]]
    offset = fcos_head_dic['proposal_generator.fcos_head.cls_tower.%d.bias'%ids[lvl][1]]
    copy_conv_gn('head.cls_convs.%d' % (lvl, ), w, b, scale, offset)


    # conv + gn
    w = fcos_head_dic['proposal_generator.fcos_head.bbox_tower.%d.weight'%ids[lvl][0]]
    b = fcos_head_dic['proposal_generator.fcos_head.bbox_tower.%d.bias'%ids[lvl][0]]
    scale = fcos_head_dic['proposal_generator.fcos_head.bbox_tower.%d.weight'%ids[lvl][1]]
    offset = fcos_head_dic['proposal_generator.fcos_head.bbox_tower.%d.bias'%ids[lvl][1]]
    copy_conv_gn('head.reg_convs.%d' % (lvl, ), w, b, scale, offset)

# 类别分支最后的conv
w = fcos_head_dic['proposal_generator.fcos_head.cls_logits.weight']
b = fcos_head_dic['proposal_generator.fcos_head.cls_logits.bias']
copy_conv('head.cls_convs.%d' % (num_convs,), w, b)

# 坐标分支最后的conv
w = fcos_head_dic['proposal_generator.fcos_head.bbox_pred.weight']
b = fcos_head_dic['proposal_generator.fcos_head.bbox_pred.bias']
copy_conv('head.reg_convs.%d' % (num_convs,), w, b)

# centerness分支最后的conv
w = fcos_head_dic['proposal_generator.fcos_head.ctrness.weight']
b = fcos_head_dic['proposal_generator.fcos_head.ctrness.bias']
copy_conv('head.ctn_conv', w, b)

# 3个scale。请注意，AdelaiDet在head部分是从小感受野到大感受野遍历，而PaddleDetection是从大感受野到小感受野遍历。所以这里scale顺序反过来。
scale_i = fcos_head_dic['proposal_generator.fcos_head.scales.0.scale']
tensor = fluid.global_scope().find_var("head.scale_%d" % 2).get_tensor()
tensor.set(scale_i, place)
scale_i = fcos_head_dic['proposal_generator.fcos_head.scales.1.scale']
tensor = fluid.global_scope().find_var("head.scale_%d" % 1).get_tensor()
tensor.set(scale_i, place)
scale_i = fcos_head_dic['proposal_generator.fcos_head.scales.2.scale']
tensor = fluid.global_scope().find_var("head.scale_%d" % 0).get_tensor()
tensor.set(scale_i, place)



fluid.io.save_persistables(exe, 'fcos_rt_r50_fpn_4x', fluid.default_startup_program())
print('\nDone.')


