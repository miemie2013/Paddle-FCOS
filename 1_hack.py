#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-09-05 18:21:37
#   Description : paddle_fcos。在PaddleDetection的tools/infer.py里插入下面代码，
#                 预测时即可保存下模型的权重文件fcos_r50_fpn_multiscale_2x.npz
#
# ================================================================
import numpy as np
from paddle import fluid


def main():
    # ...

    # 获取FCOS模型的权重
    dic = {}
    conv1_weights = np.array(fluid.global_scope().find_var('conv1_weights').get_tensor())
    bn_conv1_scale = np.array(fluid.global_scope().find_var('bn_conv1_scale').get_tensor())
    bn_conv1_offset = np.array(fluid.global_scope().find_var('bn_conv1_offset').get_tensor())
    dic['conv1_weights'] = conv1_weights
    dic['bn_conv1_scale'] = bn_conv1_scale
    dic['bn_conv1_offset'] = bn_conv1_offset
    # Resnet50
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

            branch2a_conv_weights = np.array(fluid.global_scope().find_var(conv_name1 + '_weights').get_tensor())
            branch2a_bn_scale = np.array(fluid.global_scope().find_var(bn_name1 + '_scale').get_tensor())
            branch2a_bn_offset = np.array(fluid.global_scope().find_var(bn_name1 + '_offset').get_tensor())
            dic[conv_name1 + '_weights'] = branch2a_conv_weights
            dic[bn_name1 + '_scale'] = branch2a_bn_scale
            dic[bn_name1 + '_offset'] = branch2a_bn_offset

            branch2b_conv_weights = np.array(fluid.global_scope().find_var(conv_name2 + '_weights').get_tensor())
            branch2b_bn_scale = np.array(fluid.global_scope().find_var(bn_name2 + '_scale').get_tensor())
            branch2b_bn_offset = np.array(fluid.global_scope().find_var(bn_name2 + '_offset').get_tensor())
            dic[conv_name2 + '_weights'] = branch2b_conv_weights
            dic[bn_name2 + '_scale'] = branch2b_bn_scale
            dic[bn_name2 + '_offset'] = branch2b_bn_offset

            branch2c_conv_weights = np.array(fluid.global_scope().find_var(conv_name3 + '_weights').get_tensor())
            branch2c_bn_scale = np.array(fluid.global_scope().find_var(bn_name3 + '_scale').get_tensor())
            branch2c_bn_offset = np.array(fluid.global_scope().find_var(bn_name3 + '_offset').get_tensor())
            dic[conv_name3 + '_weights'] = branch2c_conv_weights
            dic[bn_name3 + '_scale'] = branch2c_bn_scale
            dic[bn_name3 + '_offset'] = branch2c_bn_offset

            # 每个stage的第一个卷积块才有4个卷积层
            if kk == 0:
                branch1_conv_weights = np.array(fluid.global_scope().find_var(shortcut_name + '_weights').get_tensor())
                branch1_bn_scale = np.array(fluid.global_scope().find_var(shortcut_bn_name + '_scale').get_tensor())
                branch1_bn_offset = np.array(fluid.global_scope().find_var(shortcut_bn_name + '_offset').get_tensor())
                dic[shortcut_name + '_weights'] = branch1_conv_weights
                dic[shortcut_bn_name + '_scale'] = branch1_bn_scale
                dic[shortcut_bn_name + '_offset'] = branch1_bn_offset
    # fpn, 8个卷积层
    fpn_inner_res5_sum_w = np.array(fluid.global_scope().find_var('fpn_inner_res5_sum_w').get_tensor())
    fpn_inner_res5_sum_b = np.array(fluid.global_scope().find_var('fpn_inner_res5_sum_b').get_tensor())
    dic['fpn_inner_res5_sum_w'] = fpn_inner_res5_sum_w
    dic['fpn_inner_res5_sum_b'] = fpn_inner_res5_sum_b

    fpn_inner_res4_sum_lateral_w = np.array(fluid.global_scope().find_var('fpn_inner_res4_sum_lateral_w').get_tensor())
    fpn_inner_res4_sum_lateral_b = np.array(fluid.global_scope().find_var('fpn_inner_res4_sum_lateral_b').get_tensor())
    dic['fpn_inner_res4_sum_lateral_w'] = fpn_inner_res4_sum_lateral_w
    dic['fpn_inner_res4_sum_lateral_b'] = fpn_inner_res4_sum_lateral_b

    fpn_inner_res3_sum_lateral_w = np.array(fluid.global_scope().find_var('fpn_inner_res3_sum_lateral_w').get_tensor())
    fpn_inner_res3_sum_lateral_b = np.array(fluid.global_scope().find_var('fpn_inner_res3_sum_lateral_b').get_tensor())
    dic['fpn_inner_res3_sum_lateral_w'] = fpn_inner_res3_sum_lateral_w
    dic['fpn_inner_res3_sum_lateral_b'] = fpn_inner_res3_sum_lateral_b

    fpn_res5_sum_w = np.array(fluid.global_scope().find_var('fpn_res5_sum_w').get_tensor())
    fpn_res5_sum_b = np.array(fluid.global_scope().find_var('fpn_res5_sum_b').get_tensor())
    dic['fpn_res5_sum_w'] = fpn_res5_sum_w
    dic['fpn_res5_sum_b'] = fpn_res5_sum_b

    fpn_res4_sum_w = np.array(fluid.global_scope().find_var('fpn_res4_sum_w').get_tensor())
    fpn_res4_sum_b = np.array(fluid.global_scope().find_var('fpn_res4_sum_b').get_tensor())
    dic['fpn_res4_sum_w'] = fpn_res4_sum_w
    dic['fpn_res4_sum_b'] = fpn_res4_sum_b

    fpn_res3_sum_w = np.array(fluid.global_scope().find_var('fpn_res3_sum_w').get_tensor())
    fpn_res3_sum_b = np.array(fluid.global_scope().find_var('fpn_res3_sum_b').get_tensor())
    dic['fpn_res3_sum_w'] = fpn_res3_sum_w
    dic['fpn_res3_sum_b'] = fpn_res3_sum_b

    fpn_6_w = np.array(fluid.global_scope().find_var('fpn_6_w').get_tensor())
    fpn_6_b = np.array(fluid.global_scope().find_var('fpn_6_b').get_tensor())
    dic['fpn_6_w'] = fpn_6_w
    dic['fpn_6_b'] = fpn_6_b

    fpn_7_w = np.array(fluid.global_scope().find_var('fpn_7_w').get_tensor())
    fpn_7_b = np.array(fluid.global_scope().find_var('fpn_7_b').get_tensor())
    dic['fpn_7_w'] = fpn_7_w
    dic['fpn_7_b'] = fpn_7_b

    # head
    n = 5  # 有n个输出层
    num_convs = 4
    for i in range(n):  # 遍历每个输出层
        for lvl in range(0, num_convs):
            # conv + gn
            conv_cls_name = 'fcos_head_cls_tower_conv_{}'.format(lvl)
            weight_para = np.array(fluid.global_scope().find_var(conv_cls_name + "_weights").get_tensor())
            bias_para = np.array(fluid.global_scope().find_var(conv_cls_name + "_bias").get_tensor())
            norm_name = conv_cls_name + "_norm"
            _scale = np.array(fluid.global_scope().find_var(norm_name + "_scale").get_tensor())
            _offset = np.array(fluid.global_scope().find_var(norm_name + "_offset").get_tensor())
            dic[conv_cls_name + "_weights"] = weight_para
            dic[conv_cls_name + "_bias"] = bias_para
            dic[norm_name + "_scale"] = _scale
            dic[norm_name + "_offset"] = _offset

            # conv + gn
            conv_reg_name = 'fcos_head_reg_tower_conv_{}'.format(lvl)
            weight_para = np.array(fluid.global_scope().find_var(conv_reg_name + "_weights").get_tensor())
            bias_para = np.array(fluid.global_scope().find_var(conv_reg_name + "_bias").get_tensor())
            norm_name = conv_reg_name + "_norm"
            _scale = np.array(fluid.global_scope().find_var(norm_name + "_scale").get_tensor())
            _offset = np.array(fluid.global_scope().find_var(norm_name + "_offset").get_tensor())
            dic[conv_reg_name + "_weights"] = weight_para
            dic[conv_reg_name + "_bias"] = bias_para
            dic[norm_name + "_scale"] = _scale
            dic[norm_name + "_offset"] = _offset

        # 类别分支最后的conv
        conv_cls_name = "fcos_head_cls"
        weight_para = np.array(fluid.global_scope().find_var(conv_cls_name + "_weights").get_tensor())
        bias_para = np.array(fluid.global_scope().find_var(conv_cls_name + "_bias").get_tensor())
        dic[conv_cls_name + "_weights"] = weight_para
        dic[conv_cls_name + "_bias"] = bias_para

        # 坐标分支最后的conv
        conv_reg_name = "fcos_head_reg"
        weight_para = np.array(fluid.global_scope().find_var(conv_reg_name + "_weights").get_tensor())
        bias_para = np.array(fluid.global_scope().find_var(conv_reg_name + "_bias").get_tensor())
        dic[conv_reg_name + "_weights"] = weight_para
        dic[conv_reg_name + "_bias"] = bias_para

        # centerness分支最后的conv
        conv_centerness_name = "fcos_head_centerness"
        weight_para = np.array(fluid.global_scope().find_var(conv_centerness_name + "_weights").get_tensor())
        bias_para = np.array(fluid.global_scope().find_var(conv_centerness_name + "_bias").get_tensor())
        dic[conv_centerness_name + "_weights"] = weight_para
        dic[conv_centerness_name + "_bias"] = bias_para

    # 5个scale
    fpn_names = ['fpn_7', 'fpn_6', 'fpn_res5_sum', 'fpn_res4_sum', 'fpn_res3_sum']
    for fpn_name in fpn_names:
        scale_i = np.array(fluid.global_scope().find_var("%s_scale_on_reg" % fpn_name).get_tensor())
        dic["%s_scale_on_reg" % fpn_name] = scale_i

    np.savez('fcos_r50_fpn_multiscale_2x', **dic)
    print()