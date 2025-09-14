import os,sys
os.environ["CUDA_VISIBLE_DEVICES"]='1'
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import supervision as sv
import numpy as np
from PIL import Image
#conda activate py311
import time

# 数据目录
# data_dir = '/home/tangbw/works/pys/rainbow_notebooks/notebooks/mahjong-vtacs-mexax-m4vyu-sjtd-2'
data_dir='/home/tangbw/works/datas/view_check_R4/v3/R4_AUG/images'
# output_dir='mrf_output20250704'
output_dir='mrf_output20250706'
# checkpoint_path='/home/tangbw/works/pys/molrfdetr/mrf_output20250704/checkpoint0039.pth'#first trained
checkpoint_path='/home/tangbw/works/pys/molrfdetr/mrf_output20250706/checkpoint0059.pth'#second trained
# 初始化模型
rf_model = RFDETRBase()

# model.load_checkpoint(checkpoint_path)#model is wrap of torch model, 
rf_model.model_config.pretrain_weights= checkpoint_path
rf_model.model = rf_model.get_model(rf_model.model_config)

# 记录训练开始时间
start_time = time.time()
# 训练模型
rf_model.train(
    dataset_dir=data_dir,
    output_dir=output_dir,
    epochs=60,
    batch_size=4,
    grad_accum_steps=2,
    lr=1e-4
)
# /home/tangbw/works/pys/molrfdetr/mrf_output2025070/checkpoint0039.pth
#改写成resume 模型继续接着从checkpoint0039.pth 载入 再训练40个epcoch
# 记录训练结束时间
end_time = time.time()
# 计算并打印运行时间
training_time = end_time - start_time
print(f"Training completed in {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")

"""
cd /home/tangbw/works/pys/molrfdetr
nohup python train_mol_rf.py > trainoutput0706.log   2>&1  &


"""