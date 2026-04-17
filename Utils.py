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
import rdkit
from rdkit import Chem
sys.modules['model'] = imp_model


def RunPrediction(model_path, atomdf, pairdf, atom_save_path, pair_save_path):
     
    imp_model = GTNmodel(load_model = model_path)

    # print(imp_model.args)
    # print(imp_model.params)

    pred_atomdf, pred_pairdf = imp_model.predict(atomdf, pairdf, progress=True)

    if atom_save_path is not None:
        pred_atomdf.to_parquet(atom_save_path + "atomdf.parquet")
    if pair_save_path is not None:  
        pred_pairdf.to_parquet(pair_save_path + "pairdf.parquet")

def plot_histogram(data, bins=50, x_range=None, y_range=None, title=None, xlabel=None, ylabel="Frequency", logy=False, colors=None, linestyles=None, labels=None):
 
    plt.figure()

    n = len(data)

    if colors is None:
        colors = ['red'] * n
    if linestyles is None:
        linestyles = ['-'] * n
    if labels is None:
        labels = [None] * n

    # Sanity check (important for avoiding silent bugs)
    assert len(colors) == n, "colors length must match data length"
    assert len(linestyles) == n, "linestyles length must match data length"
    assert len(labels) == n, "labels length must match data length"

    for d, c, ls, lab in zip(data, colors, linestyles, labels):
        if isinstance(d, tuple):  # precomputed np.histogram
            plt.stairs(d[0], d[1], linestyle=ls, linewidth=2, color=c, label=lab)
        else:  # raw data
            plt.hist(d, bins=bins, range=x_range, histtype='step', linestyle=ls, linewidth=2, color=c, label=lab)
    
    if y_range:
        plt.ylim(y_range)

    plt.xlabel(xlabel if xlabel else "x")
    plt.ylabel(ylabel)

    if logy:
        plt.yscale("log")

    if labels and any(labels):
        plt.legend()

    plt.tight_layout()
    plt.savefig("Plots/" + (title if title else "hist") + ".png")

def plot_scatter(x_data, y_data, title=None, xlabel=None, ylabel=None, colors=None, markers=None, labels=None, alpha=0.7, s=10, x_range=None, y_range=None):

    plt.figure()

    n = len(x_data)

    # defaults
    if colors is None:
        colors = ["red"] * n
    if markers is None:
        markers = ["o"] * n
    if labels is None:
        labels = [None] * n

    # sanity checks
    assert len(y_data) == n, "x_data and y_data must match"
    assert len(colors) == n, "colors length must match data length"
    assert len(markers) == n, "markers length must match data length"
    assert len(labels) == n, "labels length must match data length"

    for x, y, c, m, lab in zip(x_data, y_data, colors, markers, labels):
        plt.scatter(x, y, color=c, marker=m, alpha=alpha, s=s, label=lab)

    if x_range:
        plt.xlim(x_range)

    if y_range:
        plt.ylim(y_range)

    plt.xlabel(xlabel if xlabel else "x")
    plt.ylabel(ylabel if ylabel else "y")

    if labels and any(labels):
        plt.legend()

    plt.tight_layout()
    plt.savefig("Plots/" + (title if title else "scatter") + ".png")
    plt.close()

def plot_histogram2d(data, bins, title=None, xlabel=None, ylabel=None, x_range=None, y_range=None):
 
    plt.figure()

    if isinstance(data, tuple) and len(data) == 3:
        # precomputed np.histogram2d output
        H, xedges, yedges = data

    else:
        # raw x,y arrays
        x, y = data
        H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[x_range, y_range])

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

def MakeDataSet2(atom_data_path, pair_data_path, type, print_head) :

    atom_data = pd.read_parquet(atom_data_path)
    pair_data = pd.read_parquet(pair_data_path)

    #NMR_Z -- All chemical shifts and scalar coupling constants
    #NMR_A -- Chemical shifts for ¹H, ¹³C, ¹⁵N, and ¹⁹F nuclei and scalar coupling correlations for H-H, C-H, N-H, and F-C 
    #NMR_B -- Chemical shifts for ¹H and ¹³C nuclei and scalar coupling correlations for H-H and C-H

    # 1JHH             --> 0
    # 2JHH - 4JHH <2Hz --> 0
    # 2JHH - 4JHH >2Hz --> 1 (COSY)
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
        pair_data.loc[pair_data["nmr_types"].str.contains("2JHH|3JHH|4JHH", na=False)  & (abs(pair_data["coupling"])>2), "coupling_label"] = 1
        pair_data.loc[pair_data["nmr_types"].str.contains("1JCH", na=False)            & (abs(pair_data["coupling"])>2), "coupling_label"] = 2
        pair_data.loc[pair_data["nmr_types"].str.contains("2JCH|3JCH|4JCH", na=False)  & (abs(pair_data["coupling"])>2), "coupling_label"] = 3
        pair_data.loc[pair_data["nmr_types"].str.contains("1JNH", na=False)            & (abs(pair_data["coupling"])>2), "coupling_label"] = 4
        pair_data.loc[pair_data["nmr_types"].str.contains("2JNH|3JNH|4JNH", na=False)  & (abs(pair_data["coupling"])>2), "coupling_label"] = 5
        pair_data.loc[pair_data["nmr_types"].str.contains("1JFC", na=False)            & (abs(pair_data["coupling"])>2), "coupling_label"] = 7
        pair_data.loc[pair_data["nmr_types"].str.contains("2JFC|3JFC|4JFC", na=False)  & (abs(pair_data["coupling"])>2), "coupling_label"] = 8

    if type=="NMR_B" :
        atom_data.loc[(atom_data["typeint"] == 7) | (atom_data["typeint"] == 9), "shift"] = 0
        #atom_data.loc[(atom_data["typeint"] == 7) | (atom_data["typeint"] == 9), "shift_mask"] = 0
        pair_data.loc[pair_data["nmr_types"].str.contains("NH|FC", na=False), "coupling_label"] = 0

    if print_head==True :
        print(atom_data.head(100).to_string())
        print("----------------------------------------------------------")
        print(pair_data.head(100).to_string())

    return atom_data, pair_data

def build_mol_bond_order(atom_df, pair_df, BO_column, rdmol=False):
    mol = Chem.RWMol()
 
    for atom_str in atom_df['typestr']:
        mol.AddAtom(Chem.Atom(atom_str))
       
    bond_list = []
    for _, row in pair_df.iterrows():
        atom_0 = row['atom_index_0']
        atom_1 = row['atom_index_1']
        bond_order = BO_column.loc[row.name]
        if atom_0 == atom_1:
            continue
        if (atom_0, atom_1) in bond_list or (atom_1, atom_0) in bond_list:
            continue
 
        if bond_order == 1:
            bond_type = Chem.BondType.SINGLE
        elif bond_order == 2:
            bond_type = Chem.BondType.DOUBLE
        elif bond_order == 3:
            bond_type = Chem.BondType.TRIPLE
        elif bond_order == 4:
            bond_type = Chem.BondType.AROMATIC
        else:
            continue
 
        mol.AddBond(atom_0, atom_1, bond_type)
        bond_list.append((atom_0, atom_1))
   
    final_mol = mol.GetMol()
 
    try:
        if not rdmol:
            return Chem.MolToSmiles(final_mol, canonical=True)
        else:
            return final_mol
    except:
        None

