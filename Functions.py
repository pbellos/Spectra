import sys
import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from imp_core_pyg.model.GTN_modules.graph_input import make_graph_df
from imp_core_pyg.model.gtn_model import GTNmodel
import imp_core_pyg.model as imp_model
import pandas as pd
from EMS.EMS import EMS
from EMS.modules.dataframe_generation.dataframe_parse import make_atoms_df, make_pairs_df
import inv_predict
import matplotlib.pyplot as plt
sys.modules['model'] = imp_model


def RunPrediction(model_path, atomdf, pairdf, atom_save_path, pair_save_path):
     
    imp_model = GTNmodel(load_model = model_path)

    pred_atomdf, pred_pairdf = imp_model.predict(atomdf, pairdf, progress=True)

    if atom_save_path is not None:
        pred_atomdf.to_parquet(atom_save_path + "atomdf.parquet")
    if pair_save_path is not None:  
        pred_pairdf.to_parquet(pair_save_path + "pairdf.parquet")

def plot_histogram(data, bins=50, x_range=None, y_range=None, title=None, xlabel=None, ylabel="Frequency", logy=False, color='red', histtype='step', linestyle='-',newFigure=True):

    if newFigure :
        plt.figure()
    plt.hist(data, bins=bins, range=x_range, color=color, histtype='step', linestyle=linestyle, linewidth=2, label=title)
    
    if y_range:
        plt.ylim(y_range)

    plt.xlabel(xlabel if xlabel else "x" )
    plt.ylabel(ylabel)

    if logy:
        plt.yscale("log")
   
    if newFigure :
      plt.tight_layout()
      plt.savefig("Plots/"+title+".png")

def plot_scatter( x, y, title, xlabel=None, ylabel=None, alpha=0.7, s=10, newFigure=False):
    
    if newFigure :
        plt.figure()

    plt.scatter(x, y, alpha=alpha, s=s)

    plt.xlabel(xlabel if xlabel else "x")
    plt.ylabel(ylabel if ylabel else "y")

    plt.tight_layout()
    if newFigure :
        plt.savefig(f"Plots/{title}.png")
        plt.close()

def plot_histogram2d(x, y, binsx=10, binsy=50, title=None, xlabel=None, ylabel=None, x_range=None, y_range=None):
 
    plt.figure()

    H, xedges, yedges = np.histogram2d(x, y, bins=[binsx, binsy], range=[x_range, y_range])

    # Mask zero bins
    H_masked = np.ma.masked_where(H == 0, H)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")  # masked bins -> white

    plt.pcolormesh(xedges, yedges, H_masked.T)

    plt.xlabel(xlabel if xlabel else "x")
    plt.ylabel(ylabel if ylabel else "y")
    plt.title(title)

    plt.colorbar(label="Counts")

    plt.tight_layout()
    plt.savefig(f"Plots/{title}.png")
    plt.close()

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
    #atom_data["shift_mask"] = 1
 
    if type == "NMR_A" or type == "NMR_B":
        atom_data.loc[atom_data["typeint"] ==8, "shift" ] = 0
        #atom_data.loc[atom_data["typeint"] ==8, "shift_mask" ] = 0
        pair_data.loc[pair_data["nmr_types"].str.contains("2JHH|3JHH|4JHH", na=False) & pair_data["coupling"]>1 , "coupling_label"] = 1
        pair_data.loc[pair_data["nmr_types"].str.contains("1JCH", na=False) & pair_data["coupling"]>2 , "coupling_label"] = 2
        pair_data.loc[pair_data["nmr_types"].str.contains("2JCH|3JCH|4JCH", na=False), "coupling_label"] = 3
        pair_data.loc[pair_data["nmr_types"].str.contains("1JNH", na=False), "coupling_label"] = 4
        pair_data.loc[pair_data["nmr_types"].str.contains("2JNH|3JNH|4JNH", na=False), "coupling_label"] = 5
        pair_data.loc[pair_data["nmr_types"].str.contains("1JFC", na=False), "coupling_label"] = 7
        pair_data.loc[pair_data["nmr_types"].str.contains("2JFC|3JFC|4JFC", na=False), "coupling_label"] = 8

    if type=="NMR_B" :
        atom_data.loc[(atom_data["typeint"] == 7) | (atom_data["typeint"] == 9), "shift"] = 0
        #atom_data.loc[(atom_data["typeint"] == 7) | (atom_data["typeint"] == 9), "shift_mask"] = 0
        pair_data.loc[pair_data["nmr_types"].str.contains("NH|FC", na=False), "coupling_label"] = 0

    if print_head==True :
        print(atom_data.head(50))
        print("----------------------------------------------------------")
        print(pair_data.head(50).to_string())

    return atom_data, pair_data

def MakeDataSet2(atom_data_path, pair_data_path, type, print_head) :

    atom_data = pd.read_parquet(atom_data_path)
    pair_data = pd.read_parquet(pair_data_path)

    print(atom_data.head(50))
    print("----------------------------------------------------------")
    print(pair_data.head(50).to_string())

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
    #atom_data["shift_mask"] = 1
 
    if type == "NMR_A" or type == "NMR_B":
        atom_data.loc[atom_data["typeint"] ==8, "shift" ] = 0
        #atom_data.loc[atom_data["typeint"] ==8, "shift_mask" ] = 0
        pair_data.loc[pair_data["nmr_types"].str.contains("2JHH|3JHH|4JHH", na=False) & pair_data["coupling"]>2 , "coupling_label"] = 1
        pair_data.loc[pair_data["nmr_types"].str.contains("1JCH", na=False) & pair_data["coupling"]>2 , "coupling_label"] = 2
        pair_data.loc[pair_data["nmr_types"].str.contains("2JCH|3JCH|4JCH", na=False) & pair_data["coupling"]>2, "coupling_label"] = 3
        pair_data.loc[pair_data["nmr_types"].str.contains("1JNH", na=False) & pair_data["coupling"]>2, "coupling_label"] = 4
        pair_data.loc[pair_data["nmr_types"].str.contains("2JNH|3JNH|4JNH", na=False) & pair_data["coupling"]>2, "coupling_label"] = 5
        pair_data.loc[pair_data["nmr_types"].str.contains("1JFC", na=False) & pair_data["coupling"]>2, "coupling_label"] = 7
        pair_data.loc[pair_data["nmr_types"].str.contains("2JFC|3JFC|4JFC", na=False) & pair_data["coupling"]>2, "coupling_label"] = 8

    if type=="NMR_B" :
        atom_data.loc[(atom_data["typeint"] == 7) | (atom_data["typeint"] == 9), "shift"] = 0
        #atom_data.loc[(atom_data["typeint"] == 7) | (atom_data["typeint"] == 9), "shift_mask"] = 0
        pair_data.loc[pair_data["nmr_types"].str.contains("NH|FC", na=False), "coupling_label"] = 0

    if print_head==True :
        print(atom_data.head(50))
        print("----------------------------------------------------------")
        print(pair_data.head(50).to_string())

    return atom_data, pair_data



