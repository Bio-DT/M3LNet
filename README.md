# M3LNet
## Install environment

In our experiments, the specific configurations are as follows:
```
python=3.9
torch=2.4.0
dgl= 2.4.1
rdkit-pypi=2022.9.5
dgllife=0.3.2
scikit-learn=1.6.1
yacs=0.1.8 
comet_ml=3.53.0
torch-geometric=2.6.1
```

You can configure the environment by using the following commands:
```
conda create -n M3LNet python=3.9
conda activate M3LNet
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 

Install the packages by using the following commands:

conda create -n M3LNet python=3.9
conda activate M3LNet
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple

Install the packages by using the following commands:
pip install --pre dgl -f https://data.dgl.ai/wheels-test/torch-2.4/cu121/repo.html -i https://pypi.tuna.tsinghua.edu.cn/simple (Linux, in our experiment, we use Linux system)      pip install  dgl -f https://data.dgl.ai/wheels/cu121/repo.html -i https://pypi.tuna.tsinghua.edu.cn/simple (windows)

Please ensure that the versions of dgl and torch are the same. Please refer to the dgl website for details (The website of dgl: https://www.dgl.ai/pages/start.html)

pip install rdkit-pypi -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install dgllife -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install yacs -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install prettytable -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install comet_ml -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch-geometric -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install fairscale -i https://pypi.tuna.tsinghua.edu.cn/simple

```
## System requirements
```
We run all of our code on the Linux system. The requirements of this system are as follows:
- Operating System: Ubuntu 22.04.5 LTS
- CPU: Intel® Xeon(R) Platinum 8370C CPU @ 2.80GHz (128GB) 
- GPU: The GPU requires 80GB of memory
```

## Training model
```
(1) For the biosnap dataset, first enter "biosnap" file and extract the contents, then please run the following commands:
- conda activate M3LNet
For a random data partitioning strategy:
- python ./M3LNet_Biosnap/train.py --cfg ./M3LNet_Biosnap/configs/Drug_Random.yaml --split random    Note: "." indicates that replacing with one's own path

For a cluster data partitioning strategy:
- python ./M3LNet_Biosnap/train.py --cfg ./M3LNet_Biosnap/configs/Drug_Cluster.yaml --split cluster    Note: "." indicates that replacing with one's own path


(2) For the bindingdb dataset, first enter "bindingdb" file and extract the contents, then please run the following commands:
- conda activate M3LNet
For a random data partitioning strategy:
- python ./M3LNet/train.py --cfg ./configs/DrugBAN.yaml --split random    Note: "." indicates that replacing with one's own path

For a cluster data partitioning strategy:
- python ./M3LNet_Bindingdb/train.py --cfg ./M3LNet_Bindingdb/configs/Drug_Cluster.yaml --split cluster    Note: "." indicates that replacing with one's own path


(3) For the human dataset (cold start experiments), first enter "human" file and extract the contents, then please run the following commands:
- conda activate M3LNet
For a random data partitioning strategy:
- python ./M3LNet_Human/train.py --cfg ./M3LNet_Human/configs/Drug.yaml --split random    Note: "." indicates that replacing with one's own path

For a cluster data partitioning strategy:
- python ./M3LNet_Human/train.py --cfg ./M3LNet_Human/configs/Drug.yaml --split cold    Note: "." indicates that replacing with one's own path

```
