#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-09-05 18:21:37
#   Description : paddle_fcos
#
# ================================================================
import cv2
from collections import deque
import math
import json
import time
import threading
import datetime
import shutil
import tempfile
import paddle.fluid.layers as L
import random
import copy
import numpy as np
from collections import OrderedDict
import os
import argparse

from config import *
from model.fcos import *
from tools.cocotools import get_classes, catid2clsid, clsid2catid
from model.decode_np import Decode
from tools.cocotools import eval
from tools.data_process import data_clean, get_samples
from tools.transform import *
from pycocotools.coco import COCO

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='FCOS Training Script')
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--config', type=int, default=2,
                    choices=[0, 1, 2],
                    help='0 -- fcos_r50_fpn_multiscale_2x.py;  1 -- fcos_rt_r50_fpn_4x.py;  2 -- fcos_rt_dla34_fpn_4x.py.')
args = parser.parse_args()
config_file = args.config
use_gpu = args.use_gpu



def _load_state(path):
    if os.path.exists(path + '.pdopt'):
        # XXX another hack to ignore the optimizer state
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, os.path.basename(os.path.normpath(path)))
        shutil.copy(path + '.pdparams', dst + '.pdparams')
        state = fluid.io.load_program_state(dst)
        shutil.rmtree(tmp)
    else:
        state = fluid.io.load_program_state(path)
    return state



def multi_thread_op(i, samples, decodeImage, context, with_mixup, mixupImage,
                     photometricDistort, randomFlipImage, normalizeImage, resizeImage, permute):
    samples[i] = decodeImage(samples[i], context)
    if with_mixup:
        samples[i] = mixupImage(samples[i], context)
    samples[i] = photometricDistort(samples[i], context)
    samples[i] = randomFlipImage(samples[i], context)
    samples[i] = normalizeImage(samples[i], context)
    samples[i] = resizeImage(samples[i], context)
    samples[i] = permute(samples[i], context)


def clear_model(save_dir):
    path_dir = os.listdir(save_dir)
    it_ids = []
    for name in path_dir:
        sss = name.split('.')
        if sss[0] == '':
            continue
        if sss[0] == 'best_model':   # 不会删除最优模型
            it_id = 9999999999
        else:
            it_id = int(sss[0])
        it_ids.append(it_id)
    if len(it_ids) >= 11 * 3:
        it_id = min(it_ids)
        pdopt_path = '%s/%d.pdopt' % (save_dir, it_id)
        pdmodel_path = '%s/%d.pdmodel' % (save_dir, it_id)
        pdparams_path = '%s/%d.pdparams' % (save_dir, it_id)
        if os.path.exists(pdopt_path):
            os.remove(pdopt_path)
        if os.path.exists(pdmodel_path):
            os.remove(pdmodel_path)
        if os.path.exists(pdparams_path):
            os.remove(pdparams_path)


if __name__ == '__main__':
    cfg = None
    if config_file == 0:
        cfg = FCOS_R50_FPN_Multiscale_2x_Config()
    elif config_file == 1:
        cfg = FCOS_RT_R50_FPN_4x_Config()
    elif config_file == 2:
        cfg = FCOS_RT_DLA34_FPN_4x_Config()

    class_names = get_classes(cfg.classes_path)
    num_classes = len(class_names)

    # 步id，无需设置，会自动读。
    iter_id = 0

    # 输出几个特征图
    n_features = len(cfg.gt2FCOSTarget['downsample_ratios'])

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            # 创建模型
            Backbone = select_backbone(cfg.backbone_type)
            backbone = Backbone(**cfg.backbone)

            Fpn = select_fpn(cfg.fpn_type)
            fpn = Fpn(**cfg.fpn)

            Loss = select_loss(cfg.fcos_loss_type)
            fcos_loss = Loss(**cfg.fcos_loss)

            Head = select_head(cfg.head_type)
            head = Head(num_classes=num_classes, fcos_loss=fcos_loss, **cfg.head)

            fcos = FCOS(backbone, fpn, head)

            image = L.data(name='image', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')
            batch_labels0 = L.data(name='batch_labels0', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='int32')
            batch_reg_target0 = L.data(name='batch_reg_target0', shape=[-1, -1, -1, 4], append_batch_size=False, dtype='float32')
            batch_centerness0 = L.data(name='batch_centerness0', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='float32')
            batch_labels1 = L.data(name='batch_labels1', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='int32')
            batch_reg_target1 = L.data(name='batch_reg_target1', shape=[-1, -1, -1, 4], append_batch_size=False, dtype='float32')
            batch_centerness1 = L.data(name='batch_centerness1', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='float32')
            batch_labels2 = L.data(name='batch_labels2', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='int32')
            batch_reg_target2 = L.data(name='batch_reg_target2', shape=[-1, -1, -1, 4], append_batch_size=False, dtype='float32')
            batch_centerness2 = L.data(name='batch_centerness2', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='float32')
            if n_features == 5:
                batch_labels3 = L.data(name='batch_labels3', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='int32')
                batch_reg_target3 = L.data(name='batch_reg_target3', shape=[-1, -1, -1, 4], append_batch_size=False, dtype='float32')
                batch_centerness3 = L.data(name='batch_centerness3', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='float32')
                batch_labels4 = L.data(name='batch_labels4', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='int32')
                batch_reg_target4 = L.data(name='batch_reg_target4', shape=[-1, -1, -1, 4], append_batch_size=False, dtype='float32')
                batch_centerness4 = L.data(name='batch_centerness4', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='float32')
            if n_features == 3:
                tag_labels = [batch_labels0, batch_labels1, batch_labels2]
                tag_bboxes = [batch_reg_target0, batch_reg_target1, batch_reg_target2]
                tag_center = [batch_centerness0, batch_centerness1, batch_centerness2]
            elif n_features == 5:
                tag_labels = [batch_labels0, batch_labels1, batch_labels2, batch_labels3, batch_labels4]
                tag_bboxes = [batch_reg_target0, batch_reg_target1, batch_reg_target2, batch_reg_target3, batch_reg_target4]
                tag_center = [batch_centerness0, batch_centerness1, batch_centerness2, batch_centerness3, batch_centerness4]
            losses = fcos(image, None, eval=False, tag_labels=tag_labels, tag_bboxes=tag_bboxes, tag_centerness=tag_center)
            loss_centerness = losses['loss_centerness']
            loss_cls = losses['loss_cls']
            loss_box = losses['loss_box']
            all_loss = loss_cls + loss_box + loss_centerness

            optimizer = fluid.optimizer.Adam(learning_rate=cfg.train_cfg['lr'])
            optimizer.minimize(all_loss)

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

    # 参数随机初始化
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    compiled_eval_prog = fluid.compiler.CompiledProgram(eval_prog)
    _decode = Decode(exe, compiled_eval_prog, class_names, cfg, for_test=False)

    # 加载权重
    if cfg.train_cfg['model_path'] is not None:
        # 加载参数, 跳过形状不匹配的。
        ignore_set = set()   # 形状对不上的可训练参数的名字
        state = _load_state(cfg.train_cfg['model_path'])   # 这是这个模型的所有参数名和参数值的字典。

        all_var_shape = {}
        for block in train_prog.blocks:
            for param in block.all_parameters():
                all_var_shape[param.name] = param.shape
        ignore_set.update([
            name for name, shape in all_var_shape.items()
            if name in state and shape != state[name].shape
        ])
        if len(ignore_set) > 0:
            for k in ignore_set:
                if k in state:
                    logger.warning('variable {} not used'.format(k))
                    del state[k]
        fluid.io.set_program_state(train_prog, state)
        strs = cfg.train_cfg['model_path'].split('weights/')
        if len(strs) == 2:
            iter_id = int(strs[1])

    # 种类id
    _catid2clsid = copy.deepcopy(catid2clsid)
    _clsid2catid = copy.deepcopy(clsid2catid)
    if num_classes != 80:   # 如果不是COCO数据集，而是自定义数据集
        _catid2clsid = {}
        _clsid2catid = {}
        for k in range(num_classes):
            _catid2clsid[k] = k
            _clsid2catid[k] = k
    # 训练集
    train_dataset = COCO(cfg.train_path)
    train_img_ids = train_dataset.getImgIds()
    train_records = data_clean(train_dataset, train_img_ids, _catid2clsid, cfg.train_pre_path)
    num_train = len(train_records)
    train_indexes = [i for i in range(num_train)]
    # 验证集
    val_dataset = COCO(cfg.val_path)
    val_img_ids = val_dataset.getImgIds()
    val_images = []   # 只跑有gt的图片，跟随PaddleDetection
    for img_id in val_img_ids:
        ins_anno_ids = val_dataset.getAnnIds(imgIds=img_id, iscrowd=False)   # 读取这张图片所有标注anno的id
        if len(ins_anno_ids) == 0:
            continue
        img_anno = val_dataset.loadImgs(img_id)[0]
        val_images.append(img_anno)

    batch_size = cfg.train_cfg['batch_size']
    with_mixup = cfg.decodeImage['with_mixup']
    context = cfg.context
    # 预处理
    # sample_transforms
    decodeImage = DecodeImage(**cfg.decodeImage)   # 对图片解码。最开始的一步。
    mixupImage = MixupImage()                      # mixup增强
    photometricDistort = PhotometricDistort()      # 颜色扭曲
    randomFlipImage = RandomFlipImage(**cfg.randomFlipImage)  # 随机翻转
    normalizeImage = NormalizeImage(**cfg.normalizeImage)     # 先除以255归一化，再减均值除以标准差
    resizeImage = ResizeImage(**cfg.resizeImage)   # 多尺度训练，随机选一个尺度，不破坏原始宽高比地缩放。具体见代码。
    permute = Permute(**cfg.permute)    # 图片从HWC格式变成CHW格式
    # batch_transforms
    padBatch = PadBatch(**cfg.padBatch)    # 由于ResizeImage()的机制特殊，这一批所有的图片的尺度不一定全相等，所以这里对齐。
    gt2FCOSTarget = Gt2FCOSTarget(**cfg.gt2FCOSTarget)   # 填写target张量。

    # 保存模型的目录
    if not os.path.exists('./weights'): os.mkdir('./weights')

    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()

    # 一轮的步数。丢弃最后几个样本。
    train_steps = num_train // batch_size
    best_ap_list = [0.0, 0]  #[map, iter]
    while True:   # 无限个epoch
        # 每个epoch之前洗乱
        np.random.shuffle(train_indexes)
        for step in range(train_steps):
            iter_id += 1

            # 估计剩余时间
            start_time = end_time
            end_time = time.time()
            time_stat.append(end_time - start_time)
            time_cost = np.mean(time_stat)
            eta_sec = (cfg.train_cfg['max_iters'] - iter_id) * time_cost
            eta = str(datetime.timedelta(seconds=int(eta_sec)))

            # ==================== train ====================
            samples = get_samples(train_records, train_indexes, step, batch_size, with_mixup)
            # sample_transforms用多线程
            threads = []
            for i in range(batch_size):
                t = threading.Thread(target=multi_thread_op, args=(i, samples, decodeImage, context, with_mixup, mixupImage,
                                                                   photometricDistort, randomFlipImage, normalizeImage, resizeImage, permute))
                threads.append(t)
                t.start()
            # 等待所有线程任务结束。
            for t in threads:
                t.join()

            # batch_transforms
            samples = padBatch(samples, context)
            samples = gt2FCOSTarget(samples, context)

            # 整理成ndarray
            _batch_images = []
            _batch_labels0 = []
            _batch_reg_target0 = []
            _batch_centerness0 = []
            _batch_labels1 = []
            _batch_reg_target1 = []
            _batch_centerness1 = []
            _batch_labels2 = []
            _batch_reg_target2 = []
            _batch_centerness2 = []
            if n_features == 5:
                _batch_labels3 = []
                _batch_reg_target3 = []
                _batch_centerness3 = []
                _batch_labels4 = []
                _batch_reg_target4 = []
                _batch_centerness4 = []
            for sample in samples:
                im = sample['image']
                _batch_images.append(np.expand_dims(im, 0))

                temp = sample['labels0']
                _batch_labels0.append(np.expand_dims(temp, 0))
                temp = sample['reg_target0']
                _batch_reg_target0.append(np.expand_dims(temp, 0))
                temp = sample['centerness0']
                _batch_centerness0.append(np.expand_dims(temp, 0))

                temp = sample['labels1']
                _batch_labels1.append(np.expand_dims(temp, 0))
                temp = sample['reg_target1']
                _batch_reg_target1.append(np.expand_dims(temp, 0))
                temp = sample['centerness1']
                _batch_centerness1.append(np.expand_dims(temp, 0))

                temp = sample['labels2']
                _batch_labels2.append(np.expand_dims(temp, 0))
                temp = sample['reg_target2']
                _batch_reg_target2.append(np.expand_dims(temp, 0))
                temp = sample['centerness2']
                _batch_centerness2.append(np.expand_dims(temp, 0))

                if n_features == 5:
                    temp = sample['labels3']
                    _batch_labels3.append(np.expand_dims(temp, 0))
                    temp = sample['reg_target3']
                    _batch_reg_target3.append(np.expand_dims(temp, 0))
                    temp = sample['centerness3']
                    _batch_centerness3.append(np.expand_dims(temp, 0))

                    temp = sample['labels4']
                    _batch_labels4.append(np.expand_dims(temp, 0))
                    temp = sample['reg_target4']
                    _batch_reg_target4.append(np.expand_dims(temp, 0))
                    temp = sample['centerness4']
                    _batch_centerness4.append(np.expand_dims(temp, 0))
            _batch_images = np.concatenate(_batch_images, 0)
            _batch_labels0 = np.concatenate(_batch_labels0, 0)
            _batch_reg_target0 = np.concatenate(_batch_reg_target0, 0)
            _batch_centerness0 = np.concatenate(_batch_centerness0, 0)
            _batch_labels1 = np.concatenate(_batch_labels1, 0)
            _batch_reg_target1 = np.concatenate(_batch_reg_target1, 0)
            _batch_centerness1 = np.concatenate(_batch_centerness1, 0)
            _batch_labels2 = np.concatenate(_batch_labels2, 0)
            _batch_reg_target2 = np.concatenate(_batch_reg_target2, 0)
            _batch_centerness2 = np.concatenate(_batch_centerness2, 0)
            if n_features == 5:
                _batch_labels3 = np.concatenate(_batch_labels3, 0)
                _batch_reg_target3 = np.concatenate(_batch_reg_target3, 0)
                _batch_centerness3 = np.concatenate(_batch_centerness3, 0)
                _batch_labels4 = np.concatenate(_batch_labels4, 0)
                _batch_reg_target4 = np.concatenate(_batch_reg_target4, 0)
                _batch_centerness4 = np.concatenate(_batch_centerness4, 0)

            if n_features == 3:
                feed_dic = {"image": _batch_images,
                            "batch_labels0": _batch_labels0,
                            "batch_reg_target0": _batch_reg_target0,
                            "batch_centerness0": _batch_centerness0,
                            "batch_labels1": _batch_labels1,
                            "batch_reg_target1": _batch_reg_target1,
                            "batch_centerness1": _batch_centerness1,
                            "batch_labels2": _batch_labels2,
                            "batch_reg_target2": _batch_reg_target2,
                            "batch_centerness2": _batch_centerness2,
                            }
            elif n_features == 5:
                feed_dic = {"image": _batch_images,
                            "batch_labels0": _batch_labels0,
                            "batch_reg_target0": _batch_reg_target0,
                            "batch_centerness0": _batch_centerness0,
                            "batch_labels1": _batch_labels1,
                            "batch_reg_target1": _batch_reg_target1,
                            "batch_centerness1": _batch_centerness1,
                            "batch_labels2": _batch_labels2,
                            "batch_reg_target2": _batch_reg_target2,
                            "batch_centerness2": _batch_centerness2,
                            "batch_labels3": _batch_labels3,
                            "batch_reg_target3": _batch_reg_target3,
                            "batch_centerness3": _batch_centerness3,
                            "batch_labels4": _batch_labels4,
                            "batch_reg_target4": _batch_reg_target4,
                            "batch_centerness4": _batch_centerness4,
                            }

            _losses = exe.run(train_prog, feed=feed_dic, fetch_list=[all_loss, loss_box, loss_cls, loss_centerness])

            _all_loss = _losses[0][0]
            _loss_box = _losses[1][0]
            _loss_cls = _losses[2][0]
            _loss_centerness = _losses[3][0]

            # ==================== log ====================
            if iter_id % 20 == 0:
                strs = 'Train iter: {}, all_loss: {:.6f}, giou_loss: {:.6f}, conf_loss: {:.6f}, cent_loss: {:.6f}, eta: {}'.format(
                    iter_id, _all_loss, _loss_box, _loss_cls, _loss_centerness, eta)
                logger.info(strs)

            # ==================== save ====================
            if iter_id % cfg.train_cfg['save_iter'] == 0:
                save_path = './weights/%d' % iter_id
                fluid.save(train_prog, save_path)
                logger.info('Save model to {}'.format(save_path))
                clear_model('weights')

            # ==================== eval ====================
            if iter_id % cfg.train_cfg['eval_iter'] == 0:
                box_ap = eval(_decode, eval_fetch_list, val_images, cfg.val_pre_path, cfg.val_path, cfg.eval_cfg['eval_batch_size'], _clsid2catid, cfg.eval_cfg['draw_image'], cfg.eval_cfg['draw_thresh'])
                logger.info("box ap: %.3f" % (box_ap[0], ))

                # 以box_ap作为标准
                ap = box_ap
                if ap[0] > best_ap_list[0]:
                    best_ap_list[0] = ap[0]
                    best_ap_list[1] = iter_id
                    save_path = './weights/best_model'
                    fluid.save(train_prog, save_path)
                    logger.info('Save model to {}'.format(save_path))
                    clear_model('weights')
                logger.info("Best test ap: {}, in iter: {}".format(best_ap_list[0], best_ap_list[1]))

            # ==================== exit ====================
            if iter_id == cfg.train_cfg['max_iters']:
                logger.info('Done.')
                exit(0)

