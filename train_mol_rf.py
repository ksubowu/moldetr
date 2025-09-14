import os,sys
os.environ["CUDA_VISIBLE_DEVICES"]='1'
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import supervision as sv
import numpy as np
from PIL import Image
import time
data_dir='/home/tangbw/works/datas/view_check_R4/v3/R4_AUG/images'
output_dir='mrf_output20250706'
checkpoint_path='/home/tangbw/works/pys/molrfdetr/mrf_output20250706/checkpoint0059.pth'
rf_model = RFDETRBase()
rf_model.model_config.pretrain_weights= checkpoint_path
rf_model.model = rf_model.get_model(rf_model.model_config)
start_time = time.time()
rf_model.train(
    dataset_dir=data_dir,
    output_dir=output_dir,
    epochs=60,
    batch_size=4,
    grad_accum_steps=2,
    lr=1e-4
)
end_time = time.time()
training_time = end_time - start_time
print(f"Training completed in {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")