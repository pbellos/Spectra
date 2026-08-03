import torch
import sys 
import glob
import os 
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt 
from imp_core_pyg.model.GTN_modules.graph_input import make_graph_df
from imp_core_pyg.model.GTN_modules.dist_env import  RANK, LOCAL_RANK, WORLD_SIZE, DEVICE 
from imp_core_pyg.model.gtn_model import GTNmodel
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler
import time
from datetime import timedelta
import argparse
import Utils as U

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def allreduce_sanity_check(device):
    t = torch.tensor(float(dist.get_rank()), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(dist.get_world_size()))
    if dist.get_rank() == 0:
        ok = abs(t.item() - expected) < 1e-3
        print(f"[sanity] all_reduce sum={t.item():.0f}, expected={expected} -> {'OK' if ok else 'BROKEN'}")

def init_process(backend):
    """
    Initialise distributed training with the provided backend.

    The world size and local rank are discovered from environment variables.
    """    
    # local_rank = int(os.environ["LOCAL_RANK"])  # must be first line
    # torch.cuda.set_device(local_rank)
    
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(seconds=60),
        world_size=WORLD_SIZE,
    )
    
    # device = torch.device(f"cuda:{local_rank}")
    # allreduce_sanity_check(device)

    
def main():

    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    if torch.cuda.is_available()==1 :
        print(torch.cuda.get_device_name(0))     ##

    #NMR_Z -- All chemical shifts and scalar coupling constants
    #NMR_A -- Chemical shifts for ¹H, ¹³C, ¹⁵N, and ¹⁹F nuclei and scalar coupling correlations for H-H, C-H, N-H, and F-C 
    #NMR_B -- Chemical shifts for ¹H and ¹³C nuclei and scalar coupling correlations for H-H and C-H

    parser = argparse.ArgumentParser(description="Main training script")
    
    #parser.add_argument("input_file", help="Path to input file")

    parser.add_argument("--target", help="distance or bond_existence", default="distance")
    parser.add_argument("--dataset_type", help="See code for description", default="NMR_Z")
    parser.add_argument("--tag", help="tag for saving the results", default="Test")
    parser.add_argument("--predict", help="TrainEvalTest", default="")
    parser.add_argument("--debug", help="Train 2 molecules to debug", default="True")

    args = parser.parse_args()

    #print(f"Input file: {args.input_file}")
    print(f"Target: {args.target}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Task tag: {args.tag}")
    print(f"Predict mode: {args.predict}")
    print(f"Debug mode: {args.debug}")

    Results_path = "/scratch/b5ao/pbellos.b5ao/Results/"

    if args.debug=="False" :
        atomdf_train_path = "/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/FCatomdf_Train.parquet"
        pairdf_train_path = "/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/FCpairdf_Train.parquet"
        atomdf_test_path  = "/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/FCatomdf_Test.parquet"
        pairdf_test_path  = "/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/FCpairdf_Test.parquet"

    else :
        atomdf_train_path = "/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/FCa_Test.parquet"
        pairdf_train_path = "/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/FCp_Test.parquet"
        atomdf_test_path  = "/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/FCa_Test.parquet"
        pairdf_test_path  = "/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/FCp_Test.parquet"

    atomdf, pairdf = U.MakeDataSet2(atomdf_train_path,pairdf_train_path,args.dataset_type,False)
    atomdf_test,pairdf_test = U.MakeDataSet2(atomdf_test_path,pairdf_test_path,args.dataset_type,False)

    # get molecules in order of appearance
    molecules = atomdf["molecule_name"].unique()
    test_molecules = atomdf_test["molecule_name"].unique()
    print(len(molecules), len(test_molecules))

    # split indices
    train_cutoff = int(0.9 * len(molecules))
    
    # molecule splits
    train_molecules = molecules[:train_cutoff]
    eval_molecules = molecules[train_cutoff:]
    
    # atom-level splits
    train_atom_df = atomdf[atomdf["molecule_name"].isin(train_molecules)]
    eval_atom_df = atomdf[atomdf["molecule_name"].isin(eval_molecules)]
    
    # pair-level splits
    train_pair_df = pairdf[pairdf["molecule_name"].isin(train_molecules)]
    eval_pair_df = pairdf[pairdf["molecule_name"].isin(eval_molecules)]

    if args.debug=="True" :
        print(train_atom_df.head(100).to_string())
        print("------------------------------------------------")
        print(train_pair_df.head(500).to_string())

    print("Datasets ready, will now build and train the model...")
    
    d_embed = 48  ##
    if args.debug=="True" :
     Epochs = 1
    else :
     Epochs = 30

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    Effective_batchsize=240//world_size

    params = {
        "task": "inverse_imp",
        "tr_epochs": Epochs, ## 3
        "n_head": 8, # must equal no. of target flags you want to predict but ideally should equal no. of total mapping keys 
        "d_embed": d_embed,
        "n_layer": 6,
        "batch_size": Effective_batchsize,
        "save_checkpoint_freq": 5,
        "final_activation": "sigmoid",   # for distances should be disabled
        #"neg_pos_ratio": 10,
        "molecule_generator": True,
        }  

    graph_attr = {
        'typeint': ('atom_types', 'int'),
        'shift': ('shift', 'float'),
        'coupling_label': ('coupling_label', 'int'),
        'nmr_types': ('nmr_types', 'int'),
        'bond_existence': ('bond_existence', 'float')
        #'distance': ('distance', 'float')
        }

    input_attr = {
        'atom_types': ('embed', 61, d_embed-1),    
        'shift': (None, None, 1),
        'coupling_label': ('embed', 100, 5),      
        'nmr_types': ('embed', 10000, d_embed-5)   #10000
        }

    model_args={'targetflag': [args.target],
                'graph_attr': graph_attr,
                'input_attr': input_attr
    }

    print("Initialsing model")
    INVmodel = GTNmodel(id="test_Pan", model_args=model_args, model_params=params)

    train_loader, _ = INVmodel.get_input((train_atom_df, train_pair_df), calculate_scaling=True,  shuffle=True)
    eval_loader,  _ = INVmodel.get_input((eval_atom_df,  eval_pair_df),  calculate_scaling=False, shuffle=True)
                                                                                                                                                                                                                     
    INVmodel.train(train_loader=train_loader, eval_loader=eval_loader, progress=True, resume=False, path=Results_path, task_name=args.tag+args.dataset_type+args.target)

    if RANK == 0:

        df = pd.read_csv(Results_path+args.tag+args.dataset_type+args.target+"/loss_metrics/"+args.target+".csv")
 
        U.plot_scatter([df["epochs"], df["epochs"]], [df["train_ml_loss"], df["eval_ml_loss"]],
                   colors=["blue", "orange"], labels=["train", "eval"], alpha=0.7, s=10,
                   title=args.tag+args.dataset_type+args.target, xlabel="Epoch", ylabel="Loss", y_range=[0, 0.1])
    
        if "Train" in args.predict:
            U.RunPrediction(Results_path+args.tag+args.dataset_type+args.target+"/"+args.tag+args.dataset_type+args.target+"_OPT_checkpoint.torch" , train_atom_df, train_pair_df, None, Results_path+args.tag+args.dataset_type+args.target+"_train_")
        if "Eval" in args.predict :
            U.RunPrediction(Results_path+args.tag+args.dataset_type+args.target+"/"+args.tag+args.dataset_type+args.target+"_OPT_checkpoint.torch" , eval_atom_df, eval_pair_df, None, Results_path+args.tag+args.dataset_type+args.target+"_eval_")
        if "Test" in args.predict :
            U.RunPrediction(Results_path+args.tag+args.dataset_type+args.target+"/"+args.tag+args.dataset_type+args.target+"_OPT_checkpoint.torch" , atomdf_test, pairdf_test, None, Results_path+args.tag+args.dataset_type+args.target+"_test_")
    

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        backend = "nccl"
        init_process(backend)  
    
    try:
        main()
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


    






#    # ---------------- test for parallel training-------------------------
#     # Keep only 19200 train and 9600 test molecules
#     molecules = atomdf["molecule_name"].unique()[:19200]
#     test_molecules = atomdf_test["molecule_name"].unique()[:9600]

#     # Filter all four dataframes
#     atomdf = atomdf[atomdf["molecule_name"].isin(molecules)]
#     pairdf = pairdf[pairdf["molecule_name"].isin(molecules)]

#     atomdf_test = atomdf_test[atomdf_test["molecule_name"].isin(test_molecules)]
#     pairdf_test = pairdf_test[pairdf_test["molecule_name"].isin(test_molecules)]

#     print(len(molecules), len(test_molecules))
#     # -------------------------------------------------------------------


    # df = pd.read_csv(Results_path+args.tag+args.dataset_type+args.target+"/loss_metrics/"+args.target+".csv")
 
    # U.plot_scatter([df["epochs"], df["epochs"]], [df["train_ml_loss"], df["eval_ml_loss"]],
    #                colors=["blue", "orange"], labels=["train", "eval"], alpha=0.7, s=10,
    #                title='trainCurve', xlabel="Epoch", ylabel="Loss")

    # if "Train" in args.predict:
    #     U.RunPrediction(Results_path+args.tag+args.dataset_type+args.target+"/"+args.tag+args.dataset_type+args.target+"_OPT_checkpoint.torch" , train_atom_df, train_pair_df, None, Results_path+args.tag+args.dataset_type+args.target+"_train_")
    # if "Eval" in args.predict :
    #     U.RunPrediction(Results_path+args.tag+args.dataset_type+args.target+"/"+args.tag+args.dataset_type+args.target+"_OPT_checkpoint.torch" , eval_atom_df, eval_pair_df, None, Results_path+args.tag+args.dataset_type+args.target+"_eval_")
    # if "Test" in args.predict :
    #     U.RunPrediction(Results_path+args.tag+args.dataset_type+args.target+"/"+args.tag+args.dataset_type+args.target+"_OPT_checkpoint.torch" , atomdf_test, pairdf_test, None, Results_path+args.tag+args.dataset_type+args.target+"_test_")
    