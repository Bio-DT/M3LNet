# M3LNet
## Install environment
You can configure the environment by using the following commands:
```
conda create -n DeepRL python=3.9
conda activate DeepRL
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

pip install torch_cluster-1.6.3+pt23cu121-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.1.2+pt23cu121-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.18+pt23cu121-cp39-cp39-linux_x86_64.whl
pip install torch_spline_conv-1.2.2+pt23cu121-cp39-cp39-linux_x86_64.whl
pip install torch-geometric
Note:Please note: Please download the corresponding .whl file from the official website. https://data.pyg.org/whl/

pip install rdkit= 2024.3.5
pip install biopython=1.83
pip install openbabel=3.1.1 
pip install plip=2.4.0

```
## System requirements
```
We run all of our code on the Linux system. The requirements of this system are as follows:
- Operating System: Ubuntu 22.04.4 LTS
- CPU: Intel® Xeon(R) Platinum 8370C CPU @ 2.80GHz (128GB) 
- GPU: NVIDIACorporationGA100 (A100 SXM480GB)
```

## Processing Data 
```
For DDA prediciton, we use two datasets, including Cdataset and Fdataset. For each dataset, the detailed processing methods are as follows:
We use preprocessing.py to process data that conforms to the model's input requirements.

For DTA prediciton, we use two datasets, including Davis and Kiba datasets. For each dataset, the detailed processing methods are as follows:
We use create_data.py to produce data formatted according to the model’s input specifications.

For 3D molecule optimizaiton, we use PDBbind dataset to pretrain model, and the dataset is downloaded from http://www.pdbbind.org.cn/.
Following this, process this dataset by preprocessing.py

Besides, we have provided all the processed data. Please extract the compressed file and place it in the same directory.
```

## Training model
```
For DDA prediction, run the following commands:
- conda activate DeepRL
- python train.py

For DTA prediction, run the following commands:
- conda activate DeepRL
- python data_processing.py
- For Davis dataset, run python train.py 0 0 0
- For Kiba dataset, run python train.py 1 1 0

For 3D molecule optimization, run the following commands:
- conda activate DeepRL
- python train.py
```
