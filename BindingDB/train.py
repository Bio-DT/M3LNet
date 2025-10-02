comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False
from models import M3LNet
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainer import Trainer
from domain_adaptator import Discriminator
import torch
import argparse
import warnings, os
import pandas as pd

"""install env
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

run commands:
For the bindingdb file, then please run the following commands:
- conda activate M3LNet
For a random data partitioning strategy:
- python ./M3LNet/train.py --cfg ./configs/DrugBAN.yaml --split random    Note: "." indicates that replacing with one's own path

For a cluster data partitioning strategy:
- python ./M3LNet_Bindingdb/train.py --cfg ./M3LNet_Bindingdb/configs/Drug_Cluster.yaml --split cluster    Note: "." indicates that replacing with one's own path

"""

###use yh_DT



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
# parser.add_argument('--cfg', required=True, help="path to config file", type=str)
# parser.add_argument('--data', required=True, type=str, metavar='TASK',
#                     help='dataset')
# parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster'])
# args = parser.parse_args()


parser = argparse.ArgumentParser(description="DTI prediction")
parser.add_argument('--cfg', help="path to config file", type=str) #required=True,   configs\\DrugBAN_DA.yaml
parser.add_argument('--data', default='bindingdb', type=str, metavar='TASK',help='dataset') #required=True; Data: bindingdb, biosnap, human
parser.add_argument('--split', type=str, metavar='S', help="split task", choices=['random', 'cluster'])
args = parser.parse_args()
#print("===========",args)
save_model = '/media/user/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DTI/M3LNet_Bindingdb/result'

def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults() ###注意修改保存的路径
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.OUTPUT_DIR)
    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'/media/user/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DTI/M3LNet_Bindingdb/datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))


    if not cfg.DA.TASK:
        train_path = os.path.join(dataFolder, "train.csv") #加载数据集
        val_path = os.path.join(dataFolder, "val.csv")
        test_path = os.path.join(dataFolder, "test.csv")
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        train_dataset = DTIDataset(df_train.index.values, df_train)  #数据集
        val_dataset = DTIDataset(df_val.index.values, df_val)
        test_dataset = DTIDataset(df_test.index.values, df_test)
    else:
        train_source_path = os.path.join(dataFolder, 'source_train.csv')
        train_target_path = os.path.join(dataFolder, 'target_train.csv')
        test_target_path = os.path.join(dataFolder, 'target_test.csv')
        df_train_source = pd.read_csv(train_source_path)
        df_train_target = pd.read_csv(train_target_path)
        df_test_target = pd.read_csv(test_target_path)

        train_dataset = DTIDataset(df_train_source.index.values, df_train_source)
        train_target_dataset = DTIDataset(df_train_target.index.values, df_train_target)
        test_target_dataset = DTIDataset(df_test_target.index.values, df_test_target)

    if cfg.COMET.USE and comet_support:
        experiment = Experiment(
            project_name=cfg.COMET.PROJECT_NAME,
            workspace=cfg.COMET.WORKSPACE,
            auto_output_logging="simple",
            log_graph=True,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
            auto_param_logging=False,
            auto_metric_logging=False
        )
        hyper_params = {
            "LR": cfg.SOLVER.LR,
            "Output_dir": cfg.RESULT.OUTPUT_DIR,
            "DA_use": cfg.DA.USE,
            "DA_task": cfg.DA.TASK,
        }
        if cfg.DA.USE:
            da_hyper_params = {
                "DA_init_epoch": cfg.DA.INIT_EPOCH,
                "Use_DA_entropy": cfg.DA.USE_ENTROPY,
                "Random_layer": cfg.DA.RANDOM_LAYER,
                "Original_random": cfg.DA.ORIGINAL_RANDOM,
                "DA_optim_lr": cfg.SOLVER.DA_LR
            }
            hyper_params.update(da_hyper_params)
        experiment.log_parameters(hyper_params)
        if cfg.COMET.TAG is not None:
            experiment.add_tag(cfg.COMET.TAG)
        experiment.set_name(f"{args.data}_{suffix}")

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}

    if not cfg.DA.USE:
        training_generator = DataLoader(train_dataset, **params)
        params['shuffle'] = False
        params['drop_last'] = False
        if not cfg.DA.TASK:
            val_generator = DataLoader(val_dataset, **params)
            test_generator = DataLoader(test_dataset, **params)
        else:
            val_generator = DataLoader(test_target_dataset, **params)
            test_generator = DataLoader(test_target_dataset, **params)
    else:
        source_generator = DataLoader(train_dataset, **params)
        target_generator = DataLoader(train_target_dataset, **params)
        n_batches = max(len(source_generator), len(target_generator))
        multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)
        params['shuffle'] = False
        params['drop_last'] = False
        val_generator = DataLoader(test_target_dataset, **params)
        test_generator = DataLoader(test_target_dataset, **params)

    model = M3LNet(**cfg).to(device)

    # ##load model
    # load_model_path = os.path.join(save_model, f"best_model_epoch_{160}.pth")
    # if os.path.exists(load_model_path):
    #     save_file_dict = torch.load(load_model_path)
    #     # 处理参数名称（如果保存时使用了多GPU训练，参数名称可能带有 "module." 前缀）
    #     new_save_file_dict = {k.replace("module.", ""): v for k, v in save_file_dict.items()}
        
    #     # 获取当前模型的 state_dict
    #     model_state_dict = model.state_dict()
        
    #     # 只加载匹配的参数
    #     for key, value in new_save_file_dict.items():
    #         if key in model_state_dict and model_state_dict[key].shape == value.shape:
    #             model_state_dict[key] = value
    #         else:
    #             print(f"Skipping {key} due to size mismatch or missing key.")
        
    #     # 加载调整后的参数
    #     model.load_state_dict(model_state_dict, strict=False)
    #     print(f"Model parameters loaded from {load_model_path}")
    # else:
    #     print(f"No saved model found at {load_model_path}, training from scratch.")



    if cfg.DA.USE:
        if cfg["DA"]["RANDOM_LAYER"]:
            domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"]).to(device)
        else:
            domain_dmm = Discriminator(input_size=cfg["DECODER"]["IN_DIM"] * cfg["DECODER"]["BINARY"],
                                       n_class=cfg["DECODER"]["BINARY"]).to(device)
        # params = list(model.parameters()) + list(domain_dmm.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)  #优化器指定
        opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    torch.backends.cudnn.benchmark = True



    if not cfg.DA.USE:   ########模型训练
        trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, opt_da=None,
                          discriminator=None, experiment=experiment, **cfg)
    else:
        trainer = Trainer(model, opt, device, multi_generator, val_generator, test_generator, opt_da=opt_da,
                          discriminator=domain_dmm, experiment=experiment, **cfg)
    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
