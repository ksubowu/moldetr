# Mol-DETR
Mol-DETR: A Lightweight Transformer Framework with Bond Anchor Points for Accurate 2D Molecular Structure Recognition and Reconstruction

## Introduction
This is a PyTorch implementation of the research: [A Lightweight Transformer Framework with Bond Anchor Points for Accurate 2D Molecular Structure Recognition and Reconstruction](https://github.com/ksubowu/moldetr/blob/master/moldetr_git.png)
![Graph abstract](https://github.com/ksubowu/moldetr/blob/master/image/Graphical%20abstract.png) 

## Environment
```
python=3.11.9
pip install -f requirements.txt
pip install rfdetr [https://github.com/roboflow/rf-detr]
```

### train 
`python train_mol_rf.py #replace the args and configs as your want` </br>
The trained model weights have been saved in `model`. </br>
### Inference to generate RDKIT molecules or Smiles
`python detr2smiles.py #replace the args and configs as need` </br>


## Acknowledgement
We thank the previous work by LW-DETR and RF-DETR teams. The code in this repository is inspired on [LW-DETR](https://github.com/Atten4Vis/LW-DETR) and [RF-DETR](https://github.com/roboflow/rf-detr)