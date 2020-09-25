

# 在本地windows运行1_pytorch2paddle.py，会生成一个yolov4文件夹，就是我们的预训练模型了。
# 把yolov4文件夹打包成zip，通过AIStudio的“创建数据集”将zip包上传。
# 仓库使用这个数据集和COCO2017数据集，就可以完成预训练模型上传了。
# 进入AIStudio，把上传的预训练模型解压：
cd ~/w*
cp ../data/data52394/FCOS_pretrained.zip ./FCOS_pretrained.zip
unzip FCOS_pretrained.zip
rm -f FCOS_pretrained.zip


# 安装依赖、解压COCO2017数据集
nvidia-smi
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


# 安装依赖、解压voc数据集
nvidia-smi
cd ~
pip install pycocotools
cd data
cd data4379
unzip pascalvoc.zip
cd ~/w*



--------------------------训练--------------------------
rm -f train.txt
nohup python train.py --config=2>> train.txt 2>&1 &

python train.py --config=2


--------------------------预测--------------------------
python demo.py --config=2


--------------------------eval--------------------------
rm -f eval.txt
nohup python eval.py --config=2>> eval.txt 2>&1 &

python eval.py --config=2



--------------------------导出--------------------------
python export_model.py --config=2


用导出后的模型预测图片：
python deploy_infer.py --model_dir inference_model --image_dir images/test/



用导出后的模型预测视频：
python deploy_infer.py --model_dir inference_model --video_file D://PycharmProjects/moviepy/che.mp4


用导出后的模型播放视频：（按esc键停止播放）
python deploy_infer.py --model_dir inference_model --play_video D://PycharmProjects/moviepy/che.mp4












