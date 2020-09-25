#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-09-05 18:21:37
#   Description : paddle_fcos。读取paddle的权重。
#
# ================================================================
from config import *
from model.fcos import FCOS



cfg = FCOS_R50_FPN_Multiscale_2x_Config()

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

dic = np.load('fcos_r50_fpn_multiscale_2x.npz')



# 获取FCOS模型的权重


# Resnet50
w = dic['conv1_weights']
scale = dic['bn_conv1_scale']
offset = dic['bn_conv1_offset']
copy_conv_af('backbone.stage1.0.conv0', w, scale, offset)


nums = [3, 4, 6, 3]
for nid, num in enumerate(nums):
    stage_name = 'res' + str(nid + 2)
    for kk in range(num):
        block_name = stage_name + chr(ord("a") + kk)
        conv_name1 = block_name + "_branch2a"
        conv_name2 = block_name + "_branch2b"
        conv_name3 = block_name + "_branch2c"
        shortcut_name = block_name + "_branch1"

        bn_name1 = 'bn' + conv_name1[3:]
        bn_name2 = 'bn' + conv_name2[3:]
        bn_name3 = 'bn' + conv_name3[3:]
        shortcut_bn_name = 'bn' + shortcut_name[3:]

        w = dic[conv_name1 + '_weights']
        scale = dic[bn_name1 + '_scale']
        offset = dic[bn_name1 + '_offset']
        copy_conv_af('backbone.stage%d.%d.conv0' % (2+nid, kk), w, scale, offset)

        w = dic[conv_name2 + '_weights']
        scale = dic[bn_name2 + '_scale']
        offset = dic[bn_name2 + '_offset']
        copy_conv_af('backbone.stage%d.%d.conv1' % (2+nid, kk), w, scale, offset)

        w = dic[conv_name3 + '_weights']
        scale = dic[bn_name3 + '_scale']
        offset = dic[bn_name3 + '_offset']
        copy_conv_af('backbone.stage%d.%d.conv2' % (2+nid, kk), w, scale, offset)

        # 每个stage的第一个卷积块才有4个卷积层
        if kk == 0:
            w = dic[shortcut_name + '_weights']
            scale = dic[shortcut_bn_name + '_scale']
            offset = dic[shortcut_bn_name + '_offset']
            copy_conv_af('backbone.stage%d.%d.conv3' % (2+nid, kk), w, scale, offset)
# fpn, 8个卷积层
w = dic['fpn_inner_res5_sum_w']
b = dic['fpn_inner_res5_sum_b']
copy_conv('fpn.s32_conv', w, b)

w = dic['fpn_inner_res4_sum_lateral_w']
b = dic['fpn_inner_res4_sum_lateral_b']
copy_conv('fpn.s16_conv', w, b)

w = dic['fpn_inner_res3_sum_lateral_w']
b = dic['fpn_inner_res3_sum_lateral_b']
copy_conv('fpn.s8_conv', w, b)

w = dic['fpn_res5_sum_w']
b = dic['fpn_res5_sum_b']
copy_conv('fpn.sc_s32_conv', w, b)

w = dic['fpn_res4_sum_w']
b = dic['fpn_res4_sum_b']
copy_conv('fpn.sc_s16_conv', w, b)

w = dic['fpn_res3_sum_w']
b = dic['fpn_res3_sum_b']
copy_conv('fpn.sc_s8_conv', w, b)

w = dic['fpn_6_w']
b = dic['fpn_6_b']
copy_conv('fpn.p6_conv', w, b)

w = dic['fpn_7_w']
b = dic['fpn_7_b']
copy_conv('fpn.p7_conv', w, b)


# head
n = 5  # 有n个输出层
num_convs = 4
for i in range(n):  # 遍历每个输出层
    for lvl in range(0, num_convs):
        # conv + gn
        conv_cls_name = 'fcos_head_cls_tower_conv_{}'.format(lvl)
        norm_name = conv_cls_name + "_norm"
        w = dic[conv_cls_name + "_weights"]
        b = dic[conv_cls_name + "_bias"]
        scale = dic[norm_name + "_scale"]
        offset = dic[norm_name + "_offset"]
        copy_conv_gn('head.cls_convs_per_feature.%d.%d' % (i, lvl), w, b, scale, offset)


        # conv + gn
        conv_reg_name = 'fcos_head_reg_tower_conv_{}'.format(lvl)
        norm_name = conv_reg_name + "_norm"
        w = dic[conv_reg_name + "_weights"]
        b = dic[conv_reg_name + "_bias"]
        scale = dic[norm_name + "_scale"]
        offset = dic[norm_name + "_offset"]
        copy_conv_gn('head.reg_convs_per_feature.%d.%d' % (i, lvl), w, b, scale, offset)

    # 类别分支最后的conv
    conv_cls_name = "fcos_head_cls"
    w = dic[conv_cls_name + "_weights"]
    b = dic[conv_cls_name + "_bias"]
    copy_conv('head.cls_convs_per_feature.%d.%d' % (i, num_convs), w, b)

    # 坐标分支最后的conv
    conv_reg_name = "fcos_head_reg"
    w = dic[conv_reg_name + "_weights"]
    b = dic[conv_reg_name + "_bias"]
    copy_conv('head.reg_convs_per_feature.%d.%d' % (i, num_convs), w, b)

    # centerness分支最后的conv
    conv_centerness_name = "fcos_head_centerness"
    w = dic[conv_centerness_name + "_weights"]
    b = dic[conv_centerness_name + "_bias"]
    copy_conv('head.ctn_convs_per_feature.%d' % (i, ), w, b)

# 5个scale
fpn_names = ['fpn_7', 'fpn_6', 'fpn_res5_sum', 'fpn_res4_sum', 'fpn_res3_sum']
i = 0
for fpn_name in fpn_names:
    scale_i = dic["%s_scale_on_reg" % fpn_name]
    tensor = fluid.global_scope().find_var("head.scale_%d" % i).get_tensor()
    tensor.set(scale_i, place)
    i += 1



fluid.io.save_persistables(exe, 'fcos_r50_fpn_multiscale_2x', fluid.default_startup_program())
print('\nDone.')


