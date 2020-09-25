# Paddle-FCOS

## 概述
Paddle-FCOS实时版，首发于AIStudio，参考自
- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- [AdelaiDet](https://github.com/aim-uofa/AdelaiDet)
- [FCOSv2论文](https://arxiv.org/pdf/2006.09214.pdf)

2020年6月出炉的论文，来自阿德莱德大学的作者团队。除了进一步升级FCOS的精度之外，作者团队还提供了FCOS的实时版↓

![](https://ai-studio-static-online.cdn.bcebos.com/e27b656b5f0e42a2826a17c15127ca27e4f09d99e0814fb2ac56f48a7a6e5de4)

![](https://ai-studio-static-online.cdn.bcebos.com/68ed353e01db4a99b9a32202cf88388f0175f04b06f447cdb1bded13ab5159b6)


图中FCOS实时版的预测速度非常抢眼，最快的模型预测速度为YOLOv3的2倍。对于此，你是不是也很心动呀？所以咩酱也迫不及待地把FCOS实时版移植到了PadllePaddle上，咩酱移植了表中画蓝框的模型。

FCOS实时版（FCOS_RT_DLA34_FPN_4x）的检测效果究竟如何呢？百闻不如一见，我们先看一个FCOS预测的视频一睹为快：

<iframe style="width:100%;height: 640px;" src="//v.qq.com/x/page/a3134bakxor.html"   webkitallowfullscreen="true" mozallowfullscreen="true" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true" autoplay=0> </iframe>

是不是很惊艳？每一帧视频中的物体大部分都被检测到。下面是咩酱移植成功之后测得的模型的mAP和FPS：


| 算法 | 骨干网络 | 图片输入大小 | mAP(COCO val2017) | FPS  |
|:------------:|:--------:|:----:|:-------:|:---------:|
| YOLOv4    | CSPDarkNet53 | (608,608)  | -  | 10.7 |
| YOLOv4    | CSPDarkNet53 | (416,416)  | -  | 19.5 |
| YOLOv4    | CSPDarkNet53 | (320,320)  | -  | 25.6 |
| YOLOv3增强版    | ResNet50-vd | (608,608)  | 0.426  | 13.5 |
| YOLOv3增强版    | ResNet50-vd | (416,416)  | 0.391  | 23.2 |
| YOLOv3增强版    | ResNet50-vd | (320,320)  | 0.352  | 29.7 |
| FCOS_R50_FPN_Multiscale_2x    | ResNet50 | (target_size=800, max_size=1333)  | 0.404  | 6.5 |
| FCOS_RT_R50_FPN_4x    | ResNet50 | (target_size=512, max_size=736)  | 0.376  | 15.8 |
| FCOS_RT_DLA34_FPN_4x    | DLA34 | (target_size=512, max_size=736)  | 0.381  | 17.5 |
| FCOS_RT_DLA34_FPN_4x    | DLA34 | (target_size=416, max_size=608)  | 0.365  | 20.8 |
| FCOS_RT_DLA34_FPN_4x    | DLA34 | (target_size=320, max_size=448)  | 0.333  | 26.2 |

**注意:**

- 测速环境为： ubuntu18.04, i5-9400F, 8GB RAM, GTX1660Ti(6GB)。
- YOLOv4和YOLOv3增强版来自我本人的另一个项目[Paddle-YOLOv4](https://aistudio.baidu.com/aistudio/projectdetail/570310) ,项目中的YOLOv3增强版是PaddleDetection中的，见[YOLOv3_ENHANCEMENT](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.3/docs/featured_model/YOLOv3_ENHANCEMENT.md)
。 FPS由Paddle-YOLOv4中export_model.py导出模型后运行python deploy_infer.py --model_dir inference_model --image_dir images/test/ 获得，包含了图片预处理和nms后处理的时间。由于原版YOLO v4使用coco trainval2014进行训练，所以mAP就不贴了。YOLOv3增强版的mAP由Paddle-YOLOv4中的eval.py跑得。
- FCOS的FPS由本项目中export_model.py导出模型后运行python deploy_infer.py --model_dir inference_model --image_dir images/test/ 获得，包含了图片预处理和nms后处理的时间。FCOS的mAP由本项目中的eval.py跑得。
- FCOS_R50_FPN_Multiscale_2x是PaddleDetection中的configs/anchor_free/fcos_r50_fpn_multiscale_2x.yml。写代码时为了保证代码的正确性也顺便带上了它。该模型官方给出的mAP为0.420，而这里为0.404，PaddleDetection中评测部分的代码我也好久没看了，可能有细微差别。（请让我偷懒一下吧！肝了十几天这个仓库了。如果你知道的话，可以告诉我呀^_^）
- FCOS_RT_R50_FPN_4x，AdelaiDet中给出的mAP为0.402，而这里为0.376，可能评测部分的代码、参数有细微差别。
- FCOS_RT_DLA34_FPN_4x，AdelaiDet中给出的mAP为0.403，而这里为0.381，可能评测部分的代码、参数有细微差别。（请让我偷懒一下吧！肝了十几天这个仓库了。如果你知道的话，可以告诉我呀^_^）
- 为什么不用V100测速？YOLOv4 (608x608)和FCOS_RT_DLA34_FPN_4x速度差距并不明显，个人能力有限，没法解释。
- 代码为什么有动态图的味道？因为我一开始是先把PaddleDetection中的configs/anchor_free/fcos_r50_fpn_multiscale_2x.yml这个模型移植到Pytorch，来学习FCOS这个算法，也把AdelaiDet中的FCOS_RT_R50_FPN_4x和FCOS_RT_DLA34_FPN_4x也加了进去。而这个仓库的代码结构几乎照搬了我写的Pytorch版FCOS，所以带有动态图的味道。（真肝帝）



读表可得：

(1)FCOS_RT_R50_FPN_4x和FCOS_RT_DLA34_FPN_4x(target_size=512, max_size=736)快于YOLOv4 (608x608)和YOLOv3增强版 (608x608)，但精度不及；

(2)YOLOv3增强版 (416x416)快于FCOS_RT_R50_FPN_4x和FCOS_RT_DLA34_FPN_4x(target_size=512, max_size=736)，精度也高一点；

(3)YOLOv3增强版 (320x320)有最快速度，精度也不错；

(4)虽然可以通过减小输入图片的大小来让FCOS_RT_DLA34_FPN_4x加速，但精度下降也是真的多；

(5)DLA34真的是一个不错的Backbone，速度是真的快，精度也不错；ResNet50-vd是更不错的Backbone。


DLA34真的是一个不错的Backbone，第一次听说DLA34是2019年CenterNet刚出的时候，那时候咩酱还是一个菜鸟，看CenterNet和DLA34的代码非常地费力，甚至当时没有成功跑成功CenterNet，因为DCN太难build了；时过境迁，今天我终于成功地把DLA34移植到飞桨平台，而飞桨原生地支持了可变形卷积，而不是额外地build了，不禁感叹一下。FCOS真的是一个不错的算法，无Anchor，直接预测格子中心点到边界框的ltrb 4个距离，简单粗暴而且有效。另外，很多实例分割算法基于FCOS，比如BlendMask、CondInst、CenterMask、PolarMask等，如果读者有兴趣往这方面发展，FCOS不得不学一下。才疏学浅，如果代码中有错误的地方或者算法与原作有出入的话欢迎提出，共同学习！



## 快速开始
(1)解压3个预训练模型
```
cd ~/w*
cp ../data/data52394/FCOS_pretrained.zip ./FCOS_pretrained.zip
unzip FCOS_pretrained.zip
rm -f FCOS_pretrained.zip
```
**注意:**

- 预训练模型是复制权重后保存得到的。对于预训练模型fcos_r50_fpn_multiscale_2x，需要将1_hack.py中的代码粘贴到PaddleDetection的tools/infer.py里，运行PaddleDetection的tools/infer.py时即可保存下模型的权重文件fcos_r50_fpn_multiscale_2x.npz，然后将fcos_r50_fpn_multiscale_2x.npz复制到本项目work目录下，再运行1_PaddleDetection_fcos_r50_fpn_multiscale_2x2paddle.py得到预训练模型。对于预训练模型fcos_rt_r50_fpn_4x，需要将AdelaiDet中给出的模型FCOS_RT_MS_R_50_4x_syncbn.pth下载到本项目work目录下，再运行1_AdelaiDet_FCOS_RT_MS_R_50_4x2paddle.py得到预训练模型（因为AIStudio中不能安装Pytorch，所以需要在本地电脑中完成。）。对于预训练模型fcos_rt_dla34_fpn_4x，需要将AdelaiDet中给出的模型FCOS_RT_MS_DLA_34_4x_syncbn.pth下载到本项目work目录下，再运行1_AdelaiDet_FCOS_RT_MS_DLA_34_4x2paddle.py得到预训练模型（因为AIStudio中不能安装Pytorch，所以需要在本地电脑中完成。）。本人已经做好并上传了预训练模型，这么复杂的操作不用读者做。



(2)导出模型
```
python export_model.py --config=2
```

**注意:**

- config=2表示使用的是配置文件fcos_rt_dla34_fpn_4x.py，config=1表示使用的是配置文件fcos_rt_r50_fpn_4x.py，config=0表示使用的是配置文件fcos_r50_fpn_multiscale_2x.py；train.py、demo.py、eval.py中同样有这个命令行参数；
- 导出后的模型放在inference_model/目录下，里面有一个infer_cfg.yml配置文件，是导出后的模型专用的。如果你要修改target_size、max_size、draw_threshold等参数，直接编辑它并保存，不需要重新导出模型。（用法和PaddleDetection中的差不多）


(3)使用模型预测图片、获取FPS
```
python deploy_infer.py --model_dir inference_model --image_dir images/test/
```
是不是很简单？


## 训练

如果你需要训练COCO2017数据集，那么需要先解压数据集
```
cd ~
pip install pycocotools
cd data
cd data7122
unzip ann*.zip
unzip val*.zip
unzip tes*.zip
unzip image_info*.zip
unzip train*.zip
cd ~/w*
```

然后输入下面命令训练（后台运行）

```
rm -f train.txt
nohup python train.py --config=2>> train.txt 2>&1 &
```
或者（前台运行）
```
python train.py --config=2
```

config=2表示使用的是配置文件fcos_rt_dla34_fpn_4x.py，config=1表示使用的是配置文件fcos_rt_r50_fpn_4x.py，config=0表示使用的是配置文件fcos_r50_fpn_multiscale_2x.py。
通过修改config/目录下的配置文件来进行更换数据集、更改超参数以及训练参数。
训练时默认每5000步计算一次验证集的mAP。或者运行eval.py评估指定模型的mAP。该mAP是val集的结果。

训练时如果发现mAP很稳定了，就停掉，修改学习率为原来的十分之一，接着继续训练，mAP可能还会再上升。暂时是这样手动操作。

## 训练自定义数据集
自带的voc2012数据集是一个很好的例子。

将自己数据集的txt注解文件放到annotation目录下，txt注解文件的格式如下：
```
xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20
xxx.jpg 48,240,195,371,11 8,12,352,498,14
# 图片名 物体1左上角x坐标,物体1左上角y坐标,物体1右下角x坐标,物体1右下角y坐标,物体1类别id 物体2左上角x坐标,物体2左上角y坐标,物体2右下角x坐标,物体2右下角y坐标,物体2类别id ...
```
运行1_txt2json.py会在annotation_json目录下生成两个coco注解风格的json注解文件，这是train.py支持的注解文件格式。
在config/下的相应配置文件里修改train_path、val_path、classes_path、train_pre_path、val_pre_path这5个变量（自带的voc2012数据集直接解除注释就ok了）就可以开始训练自己的数据集了。
如果需要跑demo.py、eval.py，与数据集有关的变量也需要修改一下，应该很容易看懂。

## 评估
```
python eval.py --config=2
```
该mAP是val集的结果。

## 预测
```
python demo.py --config=2
```

## 导出
```
python export_model.py --config=2
```
关于导出的参数请看export_model.py中的注释。导出后的模型默认存放在inference_model目录下，带有一个配置文件infer_cfg.yml。

用导出后的模型预测图片：
```
python deploy_infer.py --model_dir inference_model --image_dir images/test/
```

用导出后的模型预测视频：
```
python deploy_infer.py --model_dir inference_model --video_file D://PycharmProjects/moviepy/dddd.mp4
```

用导出后的模型播放视频：（按esc键停止播放）
```
python deploy_infer.py --model_dir inference_model --play_video D://PycharmProjects/moviepy/dddd.mp4
```



## 传送门
cv算法交流q群：645796480

但是关于项目的疑问尽量在评论上提，避免重复解答。



