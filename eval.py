#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-09-05 18:21:37
#   Description : paddle_fcos
#
# ================================================================
import copy

from config import *
from tools.cocotools import get_classes, catid2clsid, clsid2catid
import json
import os
import argparse

from tools.cocotools import eval
from model.decode_np import Decode
from model.fcos import FCOS
from tools.cocotools import get_classes

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='FCOS Eval Script')
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--config', type=int, default=2,
                    choices=[0, 1, 2],
                    help='0 -- fcos_r50_fpn_multiscale_2x.py;  1 -- fcos_rt_r50_fpn_4x.py;  2 -- fcos_rt_dla34_fpn_4x.py.')
args = parser.parse_args()
config_file = args.config
use_gpu = args.use_gpu

if __name__ == '__main__':
    cfg = None
    if config_file == 0:
        cfg = FCOS_R50_FPN_Multiscale_2x_Config()
    elif config_file == 1:
        cfg = FCOS_RT_R50_FPN_4x_Config()
    elif config_file == 2:
        cfg = FCOS_RT_DLA34_FPN_4x_Config()


    # 读取的模型
    model_path = cfg.eval_cfg['model_path']

    # 是否给图片画框。
    draw_image = cfg.eval_cfg['draw_image']
    draw_thresh = cfg.eval_cfg['draw_thresh']

    # 验证时的批大小
    eval_batch_size = cfg.eval_cfg['eval_batch_size']

    # 验证集图片的相对路径
    eval_pre_path = cfg.val_pre_path
    anno_file = cfg.val_path
    from pycocotools.coco import COCO
    val_dataset = COCO(anno_file)
    val_img_ids = val_dataset.getImgIds()
    images = []   # 只跑有gt的图片，跟随PaddleDetection
    for img_id in val_img_ids:
        ins_anno_ids = val_dataset.getAnnIds(imgIds=img_id, iscrowd=False)   # 读取这张图片所有标注anno的id
        if len(ins_anno_ids) == 0:
            continue
        img_anno = val_dataset.loadImgs(img_id)[0]
        images.append(img_anno)

    all_classes = get_classes(cfg.classes_path)
    num_classes = len(all_classes)


    startup_prog = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            # 创建模型
            Backbone = select_backbone(cfg.backbone_type)
            backbone = Backbone(**cfg.backbone)

            Fpn = select_fpn(cfg.fpn_type)
            fpn = Fpn(**cfg.fpn)

            Head = select_head(cfg.head_type)
            head = Head(num_classes=num_classes, fcos_loss=None, nms_cfg=cfg.nms_cfg, **cfg.head)

            fcos = FCOS(backbone, fpn, head)

            image = L.data(name='image', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')
            im_info = L.data(name='im_info', shape=[-1, 3], append_batch_size=False, dtype='float32')
            pred = fcos(image, im_info)

            eval_fetch_list = [pred]
    eval_prog = eval_prog.clone(for_test=True)
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    fluid.load(eval_prog, model_path, executor=exe)
    _decode = Decode(exe, eval_prog, all_classes, cfg, for_test=False)

    _clsid2catid = copy.deepcopy(clsid2catid)
    if num_classes != 80:   # 如果不是COCO数据集，而是自定义数据集
        _clsid2catid = {}
        for k in range(num_classes):
            _clsid2catid[k] = k
    box_ap = eval(_decode, eval_fetch_list, images, eval_pre_path, anno_file, eval_batch_size, _clsid2catid, draw_image, draw_thresh)

