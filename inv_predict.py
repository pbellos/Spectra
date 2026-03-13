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
from imp_core_pyg.model.gtn_model import GTNmodel
from torch.utils.tensorboard import SummaryWriter
import argparse
import DataVis

def MakeDataSet(atom_data, pair_data, type, print_head) :

    # 1JHH             --> 0
    # 2JHH - 4JHH <1Hz --> 0
    # 2JHH - 4JHH >1Hz --> 1 (COSY)
    # 1JCH        <2Hz --> 0
    # 1JCH        >2Hz --> 2 (HSQC)
    # 2JCH - 4JCH      --> 3 (HMBC)
    # 5JCH - 6JCH      --> 0
    # 1JNH             --> 4
    # 2JNH-4JNH        --> 5
    # 1JFC             --> 7
    # 2JFC-4JFC        --> 8

    pair_data["coupling_label"] = 0
 
    if type == "NMR_A" or type == "NMR_B":
        atom_data.loc[atom_data["typeint"] ==8, "shift" ] = 0
        pair_data.loc[pair_data["nmr_types"].str.contains("2JHH|3JHH|4JHH", na=False) & pair_data["coupling"]>1 , "coupling_label"] = 1
        pair_data.loc[pair_data["nmr_types"].str.contains("1JCH", na=False) & pair_data["coupling"]>2 , "coupling_label"] = 2
        pair_data.loc[pair_data["nmr_types"].str.contains("2JCH|3JCH|4JCH", na=False), "coupling_label"] = 3
        pair_data.loc[pair_data["nmr_types"].str.contains("1JNH", na=False), "coupling_label"] = 4
        pair_data.loc[pair_data["nmr_types"].str.contains("2JNH|3JNH|4JNH", na=False), "coupling_label"] = 5
        pair_data.loc[pair_data["nmr_types"].str.contains("1JFC", na=False), "coupling_label"] = 7
        pair_data.loc[pair_data["nmr_types"].str.contains("2JFC|3JFC|4JFC", na=False), "coupling_label"] = 8

    if type=="NMR_B" :
        atom_data.loc[(atom_data["typeint"] == 7) | (atom_data["typeint"] == 9), "shift"] = 0
        pair_data.loc[pair_data["nmr_types"].str.contains("NH|FC", na=False), "coupling_label"] = 0

    if print_head==True :
        print(atom_data.head(50))
        print("----------------------------------------------------------")
        print(pair_data.head(50).to_string())

    return atom_data, pair_data




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

    args = parser.parse_args()

    #print(f"Input file: {args.input_file}")
    print(f"Target: {args.target}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Task tag: {args.tag}")

    atomdf = pd.read_parquet("/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/atomdf_Train.parquet")
    pairdf = pd.read_parquet("/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/pairdf_Train.parquet")

    atomdf_test = pd.read_parquet("/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/atomdf_Test.parquet")
    pairdf_test = pd.read_parquet("/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/pairdf_Test.parquet")

    atomdf, pairdf = MakeDataSet(atomdf,pairdf,args.dataset_type,True)
    atomdf_test,pairdf_test = MakeDataSet(atomdf_test,pairdf_test,args.dataset_type,True)
 
    # get molecules in order of appearance
    molecules = atomdf["molecule_name"].unique()
    test_molecules = atomdf_test["molecule_name"].unique()
    print(len(molecules), len(test_molecules))
    
    # split indices
    train_cutoff = int(0.9 * len(molecules))  ## 0.0002
    
    # molecule splits
    train_molecules = molecules[:train_cutoff]
    ##train_molecules = train_molecules[1:]    ##
    eval_molecules = molecules[train_cutoff:]
    
    # atom-level splits
    train_atom_df = atomdf[atomdf["molecule_name"].isin(train_molecules)]
    eval_atom_df = atomdf[atomdf["molecule_name"].isin(eval_molecules)]
    
    # pair-level splits
    train_pair_df = pairdf[pairdf["molecule_name"].isin(train_molecules)]
    eval_pair_df = pairdf[pairdf["molecule_name"].isin(eval_molecules)]

    print("Datasets ready, will now build and train the model...")

    ##return 1
    
    Mywriter = SummaryWriter(log_dir="./")

    d_embed = 48  ##

    params = {
        "task": "inverse_imp",
        "tr_epochs": 50, ## 50
        "n_head": 8, # must equal no. of target flags you want to predict but ideally should equal no. of total mapping keys 
        "d_embed": d_embed,
        "n_layer": 6,
        "batch_size": 16,
        "save_checkpoint_freq": 5,
        "final_activation": "sigmoid",     # for distances should be disabled
        "molecule_generator": True
        }  

    graph_attr = {
        'typeint': ('atom_types', 'int'),
        'shift': ('shift', 'float'),
        'coupling_label': ('coupling_label', 'int'),
        'nmr_types': ('nmr_types', 'int'),
        'bond_existence': ('bond_existence', 'float')
        }

    # input_attr = {'atom_types': ('embed', 61, 99), 
    #               'nmr_types': ('embed', 10000, 99), 
    #               'coupling': (None, None, 1),
    #               'shift': (None, None, 1)}

    input_attr = {
        'atom_types': ('embed', 61, d_embed-1),    #61
        'shift': (None, None, 1),
        'coupling_label': ('embed', 100, 5),       # 100
        'nmr_types': ('embed', 10000, d_embed-5)   #10000
        }

    model_args={'targetflag': [args.target],
                'graph_attr': graph_attr,
                'input_attr': input_attr
    }

    print("Initialsing model")
    model = GTNmodel(id="test_Pan", 
                    model_args=model_args, 
                    model_params = params 
                    #load_model = "/home/b5ao/pbellos.b5ao/Spectra/Results/"+args.tag+args.dataset_type+args.target+"/"+args.tag+args.dataset_type+args.target+"_OPT_checkpoint.torch" 
                    )

   
    train_loader, _ = model.get_input((train_atom_df, train_pair_df), calculate_scaling=True, shuffle=True)

    eval_loader, _ = model.get_input((eval_atom_df, eval_pair_df), calculate_scaling=False, shuffle=True)
 #   eval_loader, _ = model.get_input((train_atom_df, train_pair_df), calculate_scaling=True, shuffle=True)     ## change between these two for testing  
                                                                                                                                                                                                                        
    model.train(train_loader=train_loader, eval_loader=eval_loader, progress=True, resume=False, path="/home/b5ao/pbellos.b5ao/Spectra/Results/", task_name=args.tag+args.dataset_type+args.target, writer=Mywriter)

    pred_atomdf, pred_pairdf = model.predict(train_atom_df, train_pair_df, progress=True)
    pred_pairdf.to_parquet("/home/b5ao/pbellos.b5ao/Spectra/Results/"+args.tag+args.dataset_type+args.target+"_pairdf_train.parquet")
    pred_atomdf, pred_pairdf = model.predict(eval_atom_df, eval_pair_df, progress=True)
    pred_pairdf.to_parquet("/home/b5ao/pbellos.b5ao/Spectra/Results/"+args.tag+args.dataset_type+args.target+"_pairdf_eval.parquet")
    pred_atomdf, pred_pairdf = model.predict(atomdf_test, pairdf_test, progress=True)    
    pred_pairdf.to_parquet("/home/b5ao/pbellos.b5ao/Spectra/Results/"+args.tag+args.dataset_type+args.target+"_pairdf_test.parquet")

    df = pd.read_csv("/home/b5ao/pbellos.b5ao/Spectra/Results/"+args.tag+args.dataset_type+args.target+"/loss_metrics/"+args.target+".csv")

    plt.figure()
    DataVis.plot_scatter( df["epochs"], df["train_ml_loss"], "train", xlabel="Epoch", ylabel="Loss", alpha=0.7, s=10, newFigure=False)
    DataVis.plot_scatter( df["epochs"], df["eval_ml_loss"], "eval",   xlabel="Epoch", ylabel="Loss", alpha=0.7, s=10, newFigure=False)
    plt.show()
    plt.savefig("/home/b5ao/pbellos.b5ao/Spectra/Results/"+args.tag+args.dataset_type+args.target+"/loss_metrics/"+args.target+".png")
    plt.close()

    df = df.iloc[0::5]
    best_epoch = df["eval_ml_loss"].idxmin()

    modelbest = GTNmodel(id="test_Pan", 
                model_args=model_args, 
                model_params = params, 
                load_model = "/home/b5ao/pbellos.b5ao/Spectra/Results/"+args.tag+args.dataset_type+args.target+"/"+args.tag+args.dataset_type+args.target+"_checkpoint_epoch_"+str(best_epoch)+".torch" 
                )
 
    pred_atomdf, pred_pairdf = modelbest.predict(train_atom_df, train_pair_df, progress=True)  ##
    pred_pairdf.to_parquet("/home/b5ao/pbellos.b5ao/Spectra/Results/"+args.tag+args.dataset_type+args.target+"_pairdf_train"+str(best_epoch)+".parquet") 
    pred_atomdf, pred_pairdf = modelbest.predict(eval_atom_df, eval_pair_df, progress=True)
    pred_pairdf.to_parquet("/home/b5ao/pbellos.b5ao/Spectra/Results/"+args.tag+args.dataset_type+args.target+"_pairdf_eval"+str(best_epoch)+".parquet")
    pred_atomdf, pred_pairdf = modelbest.predict(atomdf_test, pairdf_test, progress=True)
    pred_pairdf.to_parquet("/home/b5ao/pbellos.b5ao/Spectra/Results/"+args.tag+args.dataset_type+args.target+"_pairdf_test"+str(best_epoch)+".parquet")
    #pred_atomdf.to_parquet("/home/b5ao/pbellos.b5ao/Spectra/Results/"+args.tag+args.dataset_type+args.target+"_atomdf.parquet")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())