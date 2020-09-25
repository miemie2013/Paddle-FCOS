#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-09-05 18:21:37
#   Description : paddle_fcos。读取AdelaiDet FCOS_RT_MS_DLA_34_4x的权重。
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

state_dict = load_weights('FCOS_RT_MS_DLA_34_4x_syncbn.pth')
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



cfg = FCOS_RT_DLA34_FPN_4x_Config()

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

# dla34
w = backbone_dic['backbone.bottom_up.base_layer.0.weight']
scale = backbone_dic['backbone.bottom_up.base_layer.1.weight']
offset = backbone_dic['backbone.bottom_up.base_layer.1.bias']
m = backbone_dic['backbone.bottom_up.base_layer.1.running_mean']
v = backbone_dic['backbone.bottom_up.base_layer.1.running_var']
copy_conv_bn('dla.base_layer', w, scale, offset, m, v)

w = backbone_dic['backbone.bottom_up.level0.0.weight']
scale = backbone_dic['backbone.bottom_up.level0.1.weight']
offset = backbone_dic['backbone.bottom_up.level0.1.bias']
m = backbone_dic['backbone.bottom_up.level0.1.running_mean']
v = backbone_dic['backbone.bottom_up.level0.1.running_var']
copy_conv_bn('dla.level0.conv0', w, scale, offset, m, v)


w = backbone_dic['backbone.bottom_up.level1.0.weight']
scale = backbone_dic['backbone.bottom_up.level1.1.weight']
offset = backbone_dic['backbone.bottom_up.level1.1.bias']
m = backbone_dic['backbone.bottom_up.level1.1.running_mean']
v = backbone_dic['backbone.bottom_up.level1.1.running_var']
copy_conv_bn('dla.level1.conv0', w, scale, offset, m, v)


def copy_Tree(base_name, levels, in_channels, out_channels, name=''):
    if levels == 1:
        w = backbone_dic[name + '.tree1.conv1.weight']
        scale = backbone_dic[name + '.tree1.bn1.weight']
        offset = backbone_dic[name + '.tree1.bn1.bias']
        m = backbone_dic[name + '.tree1.bn1.running_mean']
        v = backbone_dic[name + '.tree1.bn1.running_var']
        copy_conv_bn(base_name+'.tree1.conv1', w, scale, offset, m, v)

        w = backbone_dic[name + '.tree1.conv2.weight']
        scale = backbone_dic[name + '.tree1.bn2.weight']
        offset = backbone_dic[name + '.tree1.bn2.bias']
        m = backbone_dic[name + '.tree1.bn2.running_mean']
        v = backbone_dic[name + '.tree1.bn2.running_var']
        copy_conv_bn(base_name+'.tree1.conv2', w, scale, offset, m, v)

        w = backbone_dic[name + '.tree2.conv1.weight']
        scale = backbone_dic[name + '.tree2.bn1.weight']
        offset = backbone_dic[name + '.tree2.bn1.bias']
        m = backbone_dic[name + '.tree2.bn1.running_mean']
        v = backbone_dic[name + '.tree2.bn1.running_var']
        copy_conv_bn(base_name+'.tree2.conv1', w, scale, offset, m, v)

        w = backbone_dic[name + '.tree2.conv2.weight']
        scale = backbone_dic[name + '.tree2.bn2.weight']
        offset = backbone_dic[name + '.tree2.bn2.bias']
        m = backbone_dic[name + '.tree2.bn2.running_mean']
        v = backbone_dic[name + '.tree2.bn2.running_var']
        copy_conv_bn(base_name+'.tree2.conv2', w, scale, offset, m, v)
    else:
        copy_Tree(base_name+'.tree1', levels - 1, in_channels, out_channels, name=name+'.tree1')
        copy_Tree(base_name+'.tree2', levels - 1, out_channels, out_channels, name=name+'.tree2')
    if levels == 1:
        w = backbone_dic[name + '.root.conv.weight']
        scale = backbone_dic[name + '.root.bn.weight']
        offset = backbone_dic[name + '.root.bn.bias']
        m = backbone_dic[name + '.root.bn.running_mean']
        v = backbone_dic[name + '.root.bn.running_var']
        copy_conv_bn(base_name+'.root.conv', w, scale, offset, m, v)
    if in_channels != out_channels:
        w = backbone_dic[name + '.project.0.weight']
        scale = backbone_dic[name + '.project.1.weight']
        offset = backbone_dic[name + '.project.1.bias']
        m = backbone_dic[name + '.project.1.running_mean']
        v = backbone_dic[name + '.project.1.running_var']
        copy_conv_bn(base_name+'.project', w, scale, offset, m, v)


levels = [1, 1, 1, 2, 2, 1]
channels = [16, 32, 64, 128, 256, 512]

copy_Tree('dla.level2', levels[2], channels[1], channels[2], 'backbone.bottom_up.level2')
copy_Tree('dla.level3', levels[3], channels[2], channels[3], 'backbone.bottom_up.level3')
copy_Tree('dla.level4', levels[4], channels[3], channels[4], 'backbone.bottom_up.level4')
copy_Tree('dla.level5', levels[5], channels[4], channels[5], 'backbone.bottom_up.level5')



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



fluid.io.save_persistables(exe, 'fcos_rt_dla34_fpn_4x', fluid.default_startup_program())
print('\nDone.')


