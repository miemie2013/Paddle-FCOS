#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-05 15:35:27
#   Description : 数据增强。凑不要脸地搬运了百度PaddleDetection的部分代码。
#
# ================================================================
import cv2
import uuid
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw

import logging
logger = logging.getLogger(__name__)


try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence


class BboxError(ValueError):
    pass


class ImageError(ValueError):
    pass


def is_poly(segm):
    assert isinstance(segm, (list, dict)), \
        "Invalid segm type: {}".format(type(segm))
    return isinstance(segm, list)



class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __str__(self):
        return str(self._id)


class DecodeImage(BaseOperator):
    def __init__(self, to_rgb=True, with_mixup=False, with_cutmix=False):
        """ Transform the image data to numpy format.
        Args:
            to_rgb (bool): whether to convert BGR to RGB
            with_mixup (bool): whether or not to mixup image and gt_bbbox/gt_score
            with_cutmix (bool): whether or not to cutmix image and gt_bbbox/gt_score
        """

        super(DecodeImage, self).__init__()
        self.to_rgb = to_rgb
        self.with_mixup = with_mixup
        self.with_cutmix = with_cutmix
        if not isinstance(self.to_rgb, bool):
            raise TypeError("{}: input type is invalid.".format(self))
        if not isinstance(self.with_mixup, bool):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """ load image if 'im_file' field is not empty but 'image' is"""
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode

        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        sample['image'] = im

        if 'h' not in sample:
            sample['h'] = im.shape[0]
        elif sample['h'] != im.shape[0]:
            logger.warn(
                "The actual image height: {} is not equal to the "
                "height: {} in annotation, and update sample['h'] by actual "
                "image height.".format(im.shape[0], sample['h']))
            sample['h'] = im.shape[0]
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        elif sample['w'] != im.shape[1]:
            logger.warn(
                "The actual image width: {} is not equal to the "
                "width: {} in annotation, and update sample['w'] by actual "
                "image width.".format(im.shape[1], sample['w']))
            sample['w'] = im.shape[1]

        # make default im_info with [h, w, 1]
        sample['im_info'] = np.array(
            [im.shape[0], im.shape[1], 1.], dtype=np.float32)

        # decode mixup image
        if self.with_mixup and 'mixup' in sample:
            self.__call__(sample['mixup'], context)

        # decode cutmix image
        if self.with_cutmix and 'cutmix' in sample:
            self.__call__(sample['cutmix'], context)

        # decode semantic label
        if 'semantic' in sample.keys() and sample['semantic'] is not None:
            sem_file = sample['semantic']
            sem = cv2.imread(sem_file, cv2.IMREAD_GRAYSCALE)
            sample['semantic'] = sem.astype('int32')

        return sample


class MixupImage(BaseOperator):
    def __init__(self, alpha=1.5, beta=1.5):
        """ Mixup image and gt_bbbox/gt_score
        Args:
            alpha (float): alpha parameter of beta distribute
            beta (float): beta parameter of beta distribute
        """
        super(MixupImage, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def _mixup_img(self, img1, img2, factor):
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += \
            img2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')

    def _concat_mask(self, mask1, mask2, gt_score1, gt_score2):
        h = max(mask1.shape[0], mask2.shape[0])
        w = max(mask1.shape[1], mask2.shape[1])
        expand_mask1 = np.zeros((h, w, mask1.shape[2]), 'float32')
        expand_mask2 = np.zeros((h, w, mask2.shape[2]), 'float32')
        expand_mask1[:mask1.shape[0], :mask1.shape[1], :] = mask1
        expand_mask2[:mask2.shape[0], :mask2.shape[1], :] = mask2
        l1 = len(gt_score1)
        l2 = len(gt_score2)
        if l2 == 0:
            return expand_mask1
        elif l1 == 0:
            return expand_mask2
        mask = np.concatenate((expand_mask1, expand_mask2), axis=-1)
        return mask

    def __call__(self, sample, context=None):
        if 'mixup' not in sample:
            return sample

        # 一定概率触发mixup
        if np.random.uniform(0., 1.) < 0.5:
            sample.pop('mixup')
            return sample

        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            sample.pop('mixup')
            return sample
        if factor <= 0.0:
            return sample['mixup']
        im = self._mixup_img(sample['image'], sample['mixup']['image'], factor)
        gt_bbox1 = sample['gt_bbox']
        gt_bbox2 = sample['mixup']['gt_bbox']
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = sample['gt_class']
        gt_class2 = sample['mixup']['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
        gt_score1 = sample['gt_score']
        gt_score2 = sample['mixup']['gt_score']
        gt_score = np.concatenate(
            (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
        # mask = self._concat_mask(sample['gt_mask'], sample['mixup']['gt_mask'], gt_score1, gt_score2)
        sample['image'] = im
        # sample['gt_mask'] = mask
        sample['gt_bbox'] = gt_bbox
        sample['gt_score'] = gt_score
        sample['gt_class'] = gt_class
        sample['h'] = im.shape[0]
        sample['w'] = im.shape[1]
        sample.pop('mixup')
        return sample


class PhotometricDistort(BaseOperator):
    def __init__(self):
        super(PhotometricDistort, self).__init__()

    def __call__(self, sample, context=None):
        im = sample['image']

        image = im.astype(np.float32)

        # RandomBrightness
        if np.random.randint(2):
            delta = 32
            delta = np.random.uniform(-delta, delta)
            image += delta

        state = np.random.randint(2)
        if state == 0:
            if np.random.randint(2):
                lower = 0.5
                upper = 1.5
                alpha = np.random.uniform(lower, upper)
                image *= alpha

        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if np.random.randint(2):
            lower = 0.5
            upper = 1.5
            image[:, :, 1] *= np.random.uniform(lower, upper)

        if np.random.randint(2):
            delta = 18.0
            image[:, :, 0] += np.random.uniform(-delta, delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        if state == 1:
            if np.random.randint(2):
                lower = 0.5
                upper = 1.5
                alpha = np.random.uniform(lower, upper)
                image *= alpha

        sample['image'] = image
        return sample


class RandomCrop(BaseOperator):
    """Random crop image and bboxes.

    Args:
        aspect_ratio (list): aspect ratio of cropped region.
            in [min, max] format.
        thresholds (list): iou thresholds for decide a valid bbox crop.
        scaling (list): ratio between a cropped region and the original image.
             in [min, max] format.
        num_attempts (int): number of tries before giving up.
        allow_no_crop (bool): allow return without actually cropping them.
        cover_all_box (bool): ensure all bboxes are covered in the final crop.
    """

    def __init__(self,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False):
        super(RandomCrop, self).__init__()
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box

    def __call__(self, sample, context=None):
        if 'gt_bbox' in sample and len(sample['gt_bbox']) == 0:
            return sample

        h = sample['h']
        w = sample['w']
        gt_bbox = sample['gt_bbox']

        # NOTE Original method attempts to generate one candidate for each
        # threshold then randomly sample one from the resulting list.
        # Here a short circuit approach is taken, i.e., randomly choose a
        # threshold and attempt to find a valid crop, and simply return the
        # first one found.
        # The probability is not exactly the same, kinda resembling the
        # "Monty Hall" problem. Actually carrying out the attempts will affect
        # observability (just like opening doors in the "Monty Hall" game).
        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == 'no_crop':
                return sample

            found = False
            for i in range(self.num_attempts):
                scale = np.random.uniform(*self.scaling)
                min_ar, max_ar = self.aspect_ratio
                aspect_ratio = np.random.uniform(
                    max(min_ar, scale**2), min(max_ar, scale**-2))
                crop_h = int(h * scale / np.sqrt(aspect_ratio))
                crop_w = int(w * scale * np.sqrt(aspect_ratio))
                crop_y = np.random.randint(0, h - crop_h)
                crop_x = np.random.randint(0, w - crop_w)
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                iou = self._iou_matrix(
                    gt_bbox, np.array(
                        [crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if self.cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(
                    gt_bbox, np.array(
                        crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                sample['image'] = self._crop_image(sample['image'], crop_box)
                # gt_mask = self._crop_image(sample['gt_mask'], crop_box)    # 掩码裁剪
                # sample['gt_mask'] = np.take(gt_mask, valid_ids, axis=-1)   # 掩码筛选
                sample['gt_bbox'] = np.take(cropped_box, valid_ids, axis=0)
                sample['gt_class'] = np.take(
                    sample['gt_class'], valid_ids, axis=0)
                sample['w'] = crop_box[2] - crop_box[0]
                sample['h'] = crop_box[3] - crop_box[1]
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], valid_ids, axis=0)
                return sample

        return sample

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_image(self, img, crop):
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2, :]

class RandomFlipImage(BaseOperator):
    def __init__(self, prob=0.5, is_normalized=False, is_mask_flip=False):
        """
        Args:
            prob (float): the probability of flipping image
            is_normalized (bool): whether the bbox scale to [0,1]
            is_mask_flip (bool): whether flip the segmentation
        """
        super(RandomFlipImage, self).__init__()
        self.prob = prob
        self.is_normalized = is_normalized
        self.is_mask_flip = is_mask_flip
        if not (isinstance(self.prob, float) and
                isinstance(self.is_normalized, bool) and
                isinstance(self.is_mask_flip, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def flip_segms(self, segms, height, width):
        def _flip_poly(poly, width):
            flipped_poly = np.array(poly)
            flipped_poly[0::2] = width - np.array(poly[0::2]) - 1
            return flipped_poly.tolist()

        def _flip_rle(rle, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[:, ::-1]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        flipped_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                flipped_segms.append([_flip_poly(poly, width) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                flipped_segms.append(_flip_rle(segm, height, width))
        return flipped_segms

    def flip_keypoint(self, gt_keypoint, width):
        for i in range(gt_keypoint.shape[1]):
            if i % 2 == 0:
                old_x = gt_keypoint[:, i].copy()
                if self.is_normalized:
                    gt_keypoint[:, i] = 1 - old_x
                else:
                    gt_keypoint[:, i] = width - old_x - 1
        return gt_keypoint

    def __call__(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """

        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            gt_bbox = sample['gt_bbox']
            im = sample['image']
            if not isinstance(im, np.ndarray):
                raise TypeError("{}: image is not a numpy array.".format(self))
            if len(im.shape) != 3:
                raise ImageError("{}: image is not 3-dimensional.".format(self))
            height, width, _ = im.shape
            if np.random.uniform(0, 1) < self.prob:
                im = im[:, ::-1, :]
                if gt_bbox.shape[0] == 0:
                    return sample
                oldx1 = gt_bbox[:, 0].copy()
                oldx2 = gt_bbox[:, 2].copy()
                if self.is_normalized:
                    gt_bbox[:, 0] = 1 - oldx2
                    gt_bbox[:, 2] = 1 - oldx1
                else:
                    gt_bbox[:, 0] = width - oldx2 - 1
                    gt_bbox[:, 2] = width - oldx1 - 1
                if gt_bbox.shape[0] != 0 and (
                        gt_bbox[:, 2] < gt_bbox[:, 0]).all():
                    m = "{}: invalid box, x2 should be greater than x1".format(
                        self)
                    raise BboxError(m)
                sample['gt_bbox'] = gt_bbox
                if self.is_mask_flip and len(sample['gt_poly']) != 0:
                    sample['gt_poly'] = self.flip_segms(sample['gt_poly'],
                                                        height, width)

                if 'gt_keypoint' in sample.keys():
                    sample['gt_keypoint'] = self.flip_keypoint(
                        sample['gt_keypoint'], width)

                if 'semantic' in sample.keys() and sample[
                        'semantic'] is not None:
                    sample['semantic'] = sample['semantic'][:, ::-1]

                sample['flipped'] = True
                sample['image'] = im
        sample = samples if batch_input else samples[0]
        return sample

class NormalizeBox(BaseOperator):
    """Transform the bounding box's coornidates to [0,1]."""

    def __init__(self):
        super(NormalizeBox, self).__init__()

    def __call__(self, sample, context):
        gt_bbox = sample['gt_bbox']
        width = sample['w']
        height = sample['h']
        for i in range(gt_bbox.shape[0]):
            gt_bbox[i][0] = gt_bbox[i][0] / width
            gt_bbox[i][1] = gt_bbox[i][1] / height
            gt_bbox[i][2] = gt_bbox[i][2] / width
            gt_bbox[i][3] = gt_bbox[i][3] / height
        sample['gt_bbox'] = gt_bbox
        return sample

class BboxXYXY2XYWH(BaseOperator):
    """
    Convert bbox XYXY format to XYWH format.
    """

    def __init__(self):
        super(BboxXYXY2XYWH, self).__init__()

    def __call__(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, :2]
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:4] / 2.
        sample['gt_bbox'] = bbox
        return sample


class NormalizeImage(BaseOperator):
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[1, 1, 1],
                 is_scale=True,
                 is_channel_first=True):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.is_channel_first = is_channel_first
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.is_scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, sample, context=None):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    im = im.astype(np.float32, copy=False)
                    if self.is_channel_first:
                        mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
                        std = np.array(self.std)[:, np.newaxis, np.newaxis]
                    else:
                        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
                        std = np.array(self.std)[np.newaxis, np.newaxis, :]
                    if self.is_scale:
                        im = im / 255.0
                    im -= mean
                    im /= std
                    sample[k] = im
        if not batch_input:
            samples = samples[0]
        return samples

class ResizeImage(BaseOperator):
    def __init__(self,
                 target_size=0,
                 max_size=0,
                 interp=cv2.INTER_LINEAR,
                 use_cv2=True):
        """
        Rescale image to the specified target size, and capped at max_size
        if max_size != 0.
        If target_size is list, selected a scale randomly as the specified
        target size.
        Args:
            target_size (int|list): the target size of image's short side,
                multi-scale training is adopted when type is list.
            max_size (int): the max size of image
            interp (int): the interpolation method
            use_cv2 (bool): use the cv2 interpolation method or use PIL
                interpolation method
        """
        super(ResizeImage, self).__init__()
        self.max_size = int(max_size)
        self.interp = int(interp)
        self.use_cv2 = use_cv2
        if not (isinstance(target_size, int) or isinstance(target_size, list)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List, now is {}".
                format(type(target_size)))
        self.target_size = target_size
        if not (isinstance(self.max_size, int) and isinstance(self.interp,
                                                              int)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if isinstance(self.target_size, list):
            # Case for multi-scale training
            selected_size = random.choice(self.target_size)   # 随机选一个尺度
        else:
            selected_size = self.target_size
        if float(im_size_min) == 0:
            raise ZeroDivisionError('{}: min size of image is 0'.format(self))
        if self.max_size != 0:
            im_scale = float(selected_size) / float(im_size_min)    # 短的边 缩放到选中的尺度
            # Prevent the biggest axis from being more than max_size
            if np.round(im_scale * im_size_max) > self.max_size:    # 如果缩放后 长的边 超过了self.max_size=1333，那就改为将 长的边 缩放到self.max_size
                im_scale = float(self.max_size) / float(im_size_max)
            im_scale_x = im_scale   # 不破坏原始宽高比地缩放
            im_scale_y = im_scale   # 不破坏原始宽高比地缩放

            resize_w = im_scale_x * float(im_shape[1])
            resize_h = im_scale_y * float(im_shape[0])
            im_info = [resize_h, resize_w, im_scale]   # 新的im_info
            if 'im_info' in sample and sample['im_info'][2] != 1.:
                sample['im_info'] = np.append(
                    list(sample['im_info']), im_info).astype(np.float32)
            else:
                sample['im_info'] = np.array(im_info).astype(np.float32)   # 修改sample['im_info']
        else:
            im_scale_x = float(selected_size) / float(im_shape[1])
            im_scale_y = float(selected_size) / float(im_shape[0])

            resize_w = selected_size
            resize_h = selected_size

        if self.use_cv2:
            im = cv2.resize(
                im,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            if 'semantic' in sample.keys() and sample['semantic'] is not None:
                semantic = sample['semantic']
                semantic = cv2.resize(
                    semantic.astype('float32'),
                    None,
                    None,
                    fx=im_scale_x,
                    fy=im_scale_y,
                    interpolation=self.interp)
                semantic = np.asarray(semantic).astype('int32')
                semantic = np.expand_dims(semantic, 0)
                sample['semantic'] = semantic
        else:
            if self.max_size != 0:
                raise TypeError(
                    'If you set max_size to cap the maximum size of image,'
                    'please set use_cv2 to True to resize the image.')
            im = im.astype('uint8')
            im = Image.fromarray(im)
            im = im.resize((int(resize_w), int(resize_h)), self.interp)
            im = np.array(im)
        sample['image'] = im
        return sample

class Permute(BaseOperator):
    def __init__(self, to_bgr=True, channel_first=True):
        """
        Change the channel.
        Args:
            to_bgr (bool): confirm whether to convert RGB to BGR
            channel_first (bool): confirm whether to change channel
        """
        super(Permute, self).__init__()
        self.to_bgr = to_bgr
        self.channel_first = channel_first
        if not (isinstance(self.to_bgr, bool) and
                isinstance(self.channel_first, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            assert 'image' in sample, "image data not found"
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    if self.channel_first:
                        im = np.swapaxes(im, 1, 2)
                        im = np.swapaxes(im, 1, 0)
                    if self.to_bgr:
                        im = im[[2, 1, 0], :, :]
                    sample[k] = im
        if not batch_input:
            samples = samples[0]
        return samples

class PadBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0, use_padded_im_info=True):
        super(PadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride
        self.use_padded_im_info = use_padded_im_info

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return samples
        max_shape = np.array([data['image'].shape for data in samples]).max(
            axis=0)    # max_shape=[3, max_h, max_w]

        if coarsest_stride > 0:
            max_shape[1] = int(   # max_h增加到最小的能被coarsest_stride=128整除的数
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(   # max_w增加到最小的能被coarsest_stride=128整除的数
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        padding_batch = []
        for data in samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im    # im贴在padding_im的左上部分实现对齐
            data['image'] = padding_im
            if self.use_padded_im_info:
                data['im_info'][:2] = max_shape[1:3]
            if 'semantic' in data.keys() and data['semantic'] is not None:
                semantic = data['semantic']
                padding_sem = np.zeros(
                    (1, max_shape[1], max_shape[2]), dtype=np.float32)
                padding_sem[:, :im_h, :im_w] = semantic
                data['semantic'] = padding_sem

        return samples


class Gt2FCOSTarget(BaseOperator):
    """
    Generate FCOS targets by groud truth data
    """

    def __init__(self,
                 object_sizes_boundary,
                 center_sampling_radius,
                 downsample_ratios,
                 norm_reg_targets=False):
        super(Gt2FCOSTarget, self).__init__()
        self.center_sampling_radius = center_sampling_radius
        self.downsample_ratios = downsample_ratios
        self.INF = np.inf
        self.object_sizes_boundary = [-1] + object_sizes_boundary + [self.INF]
        object_sizes_of_interest = []
        for i in range(len(self.object_sizes_boundary) - 1):
            object_sizes_of_interest.append([
                self.object_sizes_boundary[i], self.object_sizes_boundary[i + 1]
            ])
        self.object_sizes_of_interest = object_sizes_of_interest
        self.norm_reg_targets = norm_reg_targets

    def _compute_points(self, w, h):
        """
        compute the corresponding points in each feature map
        :param h: image height
        :param w: image width
        :return: points from all feature map
        """
        locations = []
        # 从小感受野stride=8遍历到大感受野stride=128。location.shape=[格子行数*格子列数, 2]，存放的是每个格子的中心点的坐标。格子顺序是第一行从左到右，第二行从左到右，...
        for stride in self.downsample_ratios:
            shift_x = np.arange(0, w, stride).astype(np.float32)
            shift_y = np.arange(0, h, stride).astype(np.float32)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shift_x = shift_x.flatten()
            shift_y = shift_y.flatten()
            location = np.stack([shift_x, shift_y], axis=1) + stride // 2
            locations.append(location)
        num_points_each_level = [len(location) for location in locations]   # num_points_each_level=[stride=8感受野格子数, ..., stride=128感受野格子数]
        locations = np.concatenate(locations, axis=0)
        return locations, num_points_each_level

    def _convert_xywh2xyxy(self, gt_bbox, w, h):
        """
        convert the bounding box from style xywh to xyxy
        :param gt_bbox: bounding boxes normalized into [0, 1]
        :param w: image width
        :param h: image height
        :return: bounding boxes in xyxy style
        """
        bboxes = gt_bbox.copy()
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        return bboxes

    def _check_inside_boxes_limited(self, gt_bbox, xs, ys,
                                    num_points_each_level):
        """
        check if points is within the clipped boxes
        :param gt_bbox: bounding boxes
        :param xs: horizontal coordinate of points
        :param ys: vertical coordinate of points
        :return: the mask of points is within gt_box or not
        """
        bboxes = np.reshape(   # [gt数, 4] -> [1, gt数, 4]
            gt_bbox, newshape=[1, gt_bbox.shape[0], gt_bbox.shape[1]])
        bboxes = np.tile(bboxes, reps=[xs.shape[0], 1, 1])   # [所有格子数, gt数, 4]   gt坐标
        ct_x = (bboxes[:, :, 0] + bboxes[:, :, 2]) / 2       # [所有格子数, gt数]      gt中心点x
        ct_y = (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2       # [所有格子数, gt数]      gt中心点y
        beg = 0   # 开始=0
        clipped_box = bboxes.copy()   # [所有格子数, gt数, 4]   gt坐标，限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
        for lvl, stride in enumerate(self.downsample_ratios):   # 遍历每个感受野，从 stride=8的感受野 到 stride=128的感受野
            end = beg + num_points_each_level[lvl]   # 结束=开始+这个感受野的格子数
            stride_exp = self.center_sampling_radius * stride   # stride_exp = 1.5 * 这个感受野的stride(的格子边长)
            clipped_box[beg:end, :, 0] = np.maximum(
                bboxes[beg:end, :, 0], ct_x[beg:end, :] - stride_exp)   # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 1] = np.maximum(
                bboxes[beg:end, :, 1], ct_y[beg:end, :] - stride_exp)   # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 2] = np.minimum(
                bboxes[beg:end, :, 2], ct_x[beg:end, :] + stride_exp)   # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 3] = np.minimum(
                bboxes[beg:end, :, 3], ct_y[beg:end, :] + stride_exp)   # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            beg = end
        # xs  [所有格子数, gt数]， 所有格子中心点的横坐标重复 gt数 次
        l_res = xs - clipped_box[:, :, 0]   # [所有格子数, gt数]  所有格子需要学习 gt数 个l
        r_res = clipped_box[:, :, 2] - xs   # [所有格子数, gt数]  所有格子需要学习 gt数 个r
        t_res = ys - clipped_box[:, :, 1]   # [所有格子数, gt数]  所有格子需要学习 gt数 个t
        b_res = clipped_box[:, :, 3] - ys   # [所有格子数, gt数]  所有格子需要学习 gt数 个b
        clipped_box_reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)   # [所有格子数, gt数, 4]  所有格子需要学习 gt数 个lrtb
        inside_gt_box = np.min(clipped_box_reg_targets, axis=2) > 0   # [所有格子数, gt数]  需要学习的lrtb如果都>0，表示格子被选中。即只选取中心点落在gt内的格子。
        return inside_gt_box

    def __call__(self, samples, context=None):
        assert len(self.object_sizes_of_interest) == len(self.downsample_ratios), \
            "object_sizes_of_interest', and 'downsample_ratios' should have same length."

        for sample in samples:
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            im_info = sample['im_info']
            bboxes = sample['gt_bbox']
            gt_class = sample['gt_class']
            gt_score = sample['gt_score']
            no_gt = False
            if len(bboxes) == 0:   # 如果没有gt，虚构一个gt为了后面不报错。
                no_gt = True
                bboxes = np.array([[0, 0, 100, 100]]).astype(np.float32)
                gt_class = np.array([[0]]).astype(np.int32)
                gt_score = np.array([[1]]).astype(np.float32)
                # print('nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnone')
            # bboxes的横坐标变成缩放后图片中对应物体的横坐标
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * np.floor(im_info[1]) / \
                np.floor(im_info[1] / im_info[2])
            # bboxes的纵坐标变成缩放后图片中对应物体的纵坐标
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * np.floor(im_info[0]) / \
                np.floor(im_info[0] / im_info[2])
            # calculate the locations
            h, w = sample['image'].shape[1:3]   # h w是这一批所有图片对齐后的高宽。
            points, num_points_each_level = self._compute_points(w, h)   # points是所有格子中心点的坐标，num_points_each_level=[stride=8感受野格子数, ..., stride=128感受野格子数]
            object_scale_exp = []
            for i, num_pts in enumerate(num_points_each_level):   # 遍历每个感受野格子数
                object_scale_exp.append(   # 边界self.object_sizes_of_interest[i] 重复 num_pts=格子数 次
                    np.tile(
                        np.array([self.object_sizes_of_interest[i]]),
                        reps=[num_pts, 1]))
            object_scale_exp = np.concatenate(object_scale_exp, axis=0)

            gt_area = (bboxes[:, 2] - bboxes[:, 0]) * (      # [gt数, ]   所有gt的面积
                bboxes[:, 3] - bboxes[:, 1])
            xs, ys = points[:, 0], points[:, 1]   # 所有格子中心点的横坐标、纵坐标
            xs = np.reshape(xs, newshape=[xs.shape[0], 1])   # [所有格子数, 1]
            xs = np.tile(xs, reps=[1, bboxes.shape[0]])      # [所有格子数, gt数]， 所有格子中心点的横坐标重复 gt数 次
            ys = np.reshape(ys, newshape=[ys.shape[0], 1])   # [所有格子数, 1]
            ys = np.tile(ys, reps=[1, bboxes.shape[0]])      # [所有格子数, gt数]， 所有格子中心点的纵坐标重复 gt数 次

            l_res = xs - bboxes[:, 0]   # [所有格子数, gt数] - [gt数, ] = [所有格子数, gt数]     结果是所有格子中心点的横坐标 分别减去 所有gt左上角的横坐标，即所有格子需要学习 gt数 个l
            r_res = bboxes[:, 2] - xs   # 所有格子需要学习 gt数 个r
            t_res = ys - bboxes[:, 1]   # 所有格子需要学习 gt数 个t
            b_res = bboxes[:, 3] - ys   # 所有格子需要学习 gt数 个b
            reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)   # [所有格子数, gt数, 4]   所有格子需要学习 gt数 个lrtb
            if self.center_sampling_radius > 0:
                # [所有格子数, gt数]    True表示格子中心点（锚点）落在gt内（gt是被限制边长后的gt）。
                # FCOS首先将gt框内的锚点（格子中心点）视为候选正样本，然后根据为每个金字塔等级定义的比例范围从候选中选择最终的正样本（而且是负责预测gt里面积最小的），最后那些未选择的锚点为负样本。
                # (1)第1个正负样本判断依据
                is_inside_box = self._check_inside_boxes_limited(
                    bboxes, xs, ys, num_points_each_level)
            else:
                # [所有格子数, gt数]    True表示格子中心点（锚点）落在gt内。
                # FCOS首先将gt框内的锚点（格子中心点）视为候选正样本，然后根据为每个金字塔等级定义的比例范围从候选中选择最终的正样本（而且是负责预测gt里面积最小的），最后那些未选择的锚点为负样本。
                # (1)第1个正负样本判断依据
                is_inside_box = np.min(reg_targets, axis=2) > 0
            # check if the targets is inside the corresponding level
            max_reg_targets = np.max(reg_targets, axis=2)    # [所有格子数, gt数]   所有格子需要学习 gt数 个lrtb   中的最大值
            lower_bound = np.tile(    # [所有格子数, gt数]   下限重复 gt数 次
                np.expand_dims(
                    object_scale_exp[:, 0], axis=1),
                reps=[1, max_reg_targets.shape[1]])
            high_bound = np.tile(     # [所有格子数, gt数]   上限重复 gt数 次
                np.expand_dims(
                    object_scale_exp[:, 1], axis=1),
                reps=[1, max_reg_targets.shape[1]])

            # [所有格子数, gt数]   最大回归值如果位于区间内，就为True
            # (2)第2个正负样本判断依据
            is_match_current_level = \
                (max_reg_targets > lower_bound) & \
                (max_reg_targets < high_bound)
            # [所有格子数, gt数]   所有gt的面积
            points2gtarea = np.tile(
                np.expand_dims(
                    gt_area, axis=0), reps=[xs.shape[0], 1])
            points2gtarea[is_inside_box == 0] = self.INF            # 格子中心点落在gt外的（即负样本），需要学习的面积置为无穷。     这是为了points2gtarea.min(axis=1)时，若某格子有最终正样本，那么就应该不让负样本的面积影响到判断。
            points2gtarea[is_match_current_level == 0] = self.INF   # 最大回归值如果位于区间外（即负样本），需要学习的面积置为无穷。 这是为了points2gtarea.min(axis=1)时，若某格子有最终正样本，那么就应该不让负样本的面积影响到判断。
            points2min_area = points2gtarea.min(axis=1)          # [所有格子数, ]   所有格子需要学习 gt数 个面积  中的最小值
            points2min_area_ind = points2gtarea.argmin(axis=1)   # [所有格子数, ]   所有格子需要学习 gt数 个面积  中的最小值的下标
            labels = gt_class[points2min_area_ind] + 1     # [所有格子数, 1]   所有格子需要学习 的类别id，学习的是gt中面积最小值的的类别id
            labels[points2min_area == self.INF] = 0        # [所有格子数, 1]   负样本的points2min_area肯定是self.INF，这里将负样本需要学习 的类别id 置为0
            reg_targets = reg_targets[range(xs.shape[0]), points2min_area_ind]   # [所有格子数, 4]   所有格子需要学习 的 lrtb（负责预测gt里面积最小的）
            ctn_targets = np.sqrt((reg_targets[:, [0, 2]].min(axis=1) / \
                                  reg_targets[:, [0, 2]].max(axis=1)) * \
                                  (reg_targets[:, [1, 3]].min(axis=1) / \
                                   reg_targets[:, [1, 3]].max(axis=1))).astype(np.float32)   # [所有格子数, ]  所有格子需要学习的centerness
            ctn_targets = np.reshape(
                ctn_targets, newshape=[ctn_targets.shape[0], 1])   # [所有格子数, 1]  所有格子需要学习的centerness
            ctn_targets[labels <= 0] = 0   # 负样本需要学习的centerness置为0
            pos_ind = np.nonzero(labels != 0)   # tuple=( ndarray(shape=[正样本数, ]), ndarray(shape=[正样本数, ]) )   即正样本在labels中的下标，因为labels是2维的，所以一个正样本有2个下标。
            reg_targets_pos = reg_targets[pos_ind[0], :]    # [正样本数, 4]   正样本格子需要学习 的 lrtb
            split_sections = []   # 每一个感受野 最后一个格子 在reg_targets中的位置（第一维的位置）
            beg = 0
            for lvl in range(len(num_points_each_level)):
                end = beg + num_points_each_level[lvl]
                split_sections.append(end)
                beg = end
            if no_gt:   # 如果没有gt，labels里全部置为0（背景的类别id是0）即表示所有格子都是负样本
                labels[:, :] = 0
            labels_by_level = np.split(labels, split_sections, axis=0)             # 一个list，根据split_sections切分，各个感受野的target切分开来。
            reg_targets_by_level = np.split(reg_targets, split_sections, axis=0)   # 一个list，根据split_sections切分，各个感受野的target切分开来。
            ctn_targets_by_level = np.split(ctn_targets, split_sections, axis=0)   # 一个list，根据split_sections切分，各个感受野的target切分开来。

            # 最后一步是reshape，和格子的位置对应上。
            for lvl in range(len(self.downsample_ratios)):
                grid_w = int(np.ceil(w / self.downsample_ratios[lvl]))   # 格子列数
                grid_h = int(np.ceil(h / self.downsample_ratios[lvl]))   # 格子行数
                if self.norm_reg_targets:   # 是否将reg目标归一化，配置里是True
                    sample['reg_target{}'.format(lvl)] = \
                        np.reshape(
                            reg_targets_by_level[lvl] / \
                            self.downsample_ratios[lvl],      # 归一化方式是除以格子边长（即下采样倍率）
                            newshape=[grid_h, grid_w, 4])     # reshape成[grid_h, grid_w, 4]
                else:
                    sample['reg_target{}'.format(lvl)] = np.reshape(
                        reg_targets_by_level[lvl],
                        newshape=[grid_h, grid_w, 4])
                sample['labels{}'.format(lvl)] = np.reshape(
                    labels_by_level[lvl], newshape=[grid_h, grid_w, 1])     # reshape成[grid_h, grid_w, 1]
                sample['centerness{}'.format(lvl)] = np.reshape(
                    ctn_targets_by_level[lvl], newshape=[grid_h, grid_w, 1])     # reshape成[grid_h, grid_w, 1]
        return samples




