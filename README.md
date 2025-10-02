# M3LNet
## Install environment

In our experiments, the specific configurations are as follows:
```
python=3.9
torch=2.4.0
dgl=
rdkit=
dgllife=
scikit-learn=
yacs=
comet_ml=
torch-geometric=
```

You can configure the environment by using the following commands:
```
conda create -n M3LNet python=3.9
conda activate M3LNet
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

Install the packages by using the following commands:
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
pip install rdkit-pypi
pip install dgllife
pip install -U scikit-learn
pip install yacs
pip install prettytable
pip install comet_ml
pip install torch-geometric

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
- python main.py --cfg './configs/DrugBAN.yaml' --split 'random'    Note: "." indicates that replacing with one's own path

For a cluster data partitioning strategy:
- python main.py --cfg './configs/DrugBAN_DA' --split 'cluster'    Note: "." indicates that replacing with one's own path


(2) For the bindingdb dataset, first enter "bindingdb" file and extract the contents, then please run the following commands:
- conda activate M3LNet
For a random data partitioning strategy:
- python main.py --cfg './configs/DrugBAN.yaml' --split 'random'    Note: "." indicates that replacing with one's own path

For a cluster data partitioning strategy:
- python main.py --cfg './configs/DrugBAN_DA' --split 'cluster'    Note: "." indicates that replacing with one's own path


(3) For the human dataset (cold start experiments), first enter "human" file and extract the contents, then please run the following commands:
- conda activate M3LNet
For a random data partitioning strategy:
- python main.py --cfg './configs/DrugBAN.yaml' --split 'random'    Note: "." indicates that replacing with one's own path

For a cluster data partitioning strategy:
- python main.py --cfg './configs/DrugBAN.yaml' --split 'cold'    Note: "." indicates that replacing with one's own path

```
