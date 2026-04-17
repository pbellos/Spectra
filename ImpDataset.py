import sys 
import glob
import os 
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Utils as U

class ImpDataset:

    def __init__(self, Title="Test", atom_df = None,  pair_df = None):
        self.Title = Title
        if pair_df is None:
            pass 
        elif pair_df in ["Train", "Test", "Eval"]:
            self.pair_data = pd.read_parquet(f"/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/FCpairdf_{pair_df}.parquet")
        else:
            self.pair_data = pd.read_parquet(pair_df)
        if atom_df is None:
            pass 
        elif atom_df in ["Train", "Test", "Eval"]:
            self.atom_data = pd.read_parquet(f"/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/FCatomdf_{atom_df}.parquet")
        else:
            self.atom_data = pd.read_parquet(atom_df)

        self.Metrics = {}
        self.Plots = {}

        print(f"Loaded {self.Title}")

        # print(self.pair_data.head(100).to_string())

    def AnalyzeBonds(self, parameters, metrics, plots, save_plots, print_header):
        
        bond_filter, bond_threshold = parameters[0], parameters[1]

        mask = self.pair_data["nmr_types"].str.contains(bond_filter, na=False)
        pos_mask = (self.pair_data["bond_existence"] == 1) & mask
        neg_mask = (self.pair_data["bond_existence"] == 0) & mask

        metric_funcs = {
            "Recall":           lambda: self.calculate_recall(pos_mask, bond_threshold),
            "Precision":        lambda: self.calculate_precision(pos_mask, neg_mask, bond_threshold),
            "PerfectMolecules": lambda: self.calculate_perfect_molecules(mask, bond_threshold),
            "SmilesAccuracy":   lambda: self.calculate_smiles_accuracy(bond_threshold),
        }

        plot_funcs = {
            "PBP": lambda: np.histogram(self.pair_data.loc[pos_mask, "predicted_bond_existence"], bins=40, range=(0,1)),
            "NBP": lambda: np.histogram(self.pair_data.loc[neg_mask, "predicted_bond_existence"], bins=40, range=(0,1)),
        }

        self.Metrics = {metric: metric_funcs[metric]() for metric in metrics}
        self.Plots = {plot: plot_funcs[plot]() for plot in plots}
        self.PrintMetrics(print_header)
        if save_plots==1 :
            self.save_plots()

    def AnalyzeDistances(self, parameters, metrics, plots, save_plots, print_header=True) :
 
        Max_path_lenght = parameters[0]

        mask = (self.pair_data["path_len"] <= Max_path_lenght)

        metric_funcs = {
        "DistanceMeanError":          lambda: self.calculate_distance_ME(mask),
        }

        plot_funcs = {
        "DistanceMEPerMolecule":  lambda: np.histogram(self.calculate_distance_ME_per_molecule(mask), bins=50, ),
        "DistancePredVsTrue":     lambda: np.histogram2d(self.pair_data.loc[mask, "distance"], self.pair_data.loc[mask, "predicted_distance"], bins=[40,40],range=[[0, 4], [0, 4]]),
        }

        self.Metrics = {metric: metric_funcs[metric]() for metric in metrics}
        self.Plots   = {plot:   plot_funcs[plot]()     for plot   in plots}
        self.PrintMetrics(print_header)
        if save_plots==1 :
            self.SavePlots()


    def calculate_distance_ME(self, mask):
        return (self.pair_data.loc[mask, "distance"] - self.pair_data.loc[mask, "predicted_distance"]).mean()

    def calculate_distance_ME_per_molecule(self, mask):
        error = self.pair_data.loc[mask, "distance"] - self.pair_data.loc[mask, "predicted_distance"]
        return error.groupby(self.pair_data.loc[mask, "molecule_name"]).mean().values

    def calculate_precision(self, pos_mask, neg_mask, bond_threshold):
        """
        Precision: of all bonds predicted as existing, how many actually exist?
        TP / (TP + FP)
        """
        TP = (pos_mask & (self.pair_data["predicted_bond_existence"] > bond_threshold)).sum()
        FP = (neg_mask & (self.pair_data["predicted_bond_existence"] > bond_threshold)).sum()

        return TP / (TP + FP)


    def calculate_recall(self, pos_mask, bond_threshold):
        """
        Recall: of all bonds that actually exist, how many did we predict correctly?
        TP / (TP + FN)
        """
        TP = (pos_mask & (self.pair_data["predicted_bond_existence"] > bond_threshold)).sum()
        FN = (pos_mask & (self.pair_data["predicted_bond_existence"] <= bond_threshold)).sum()

        return TP / (TP + FN)


    def calculate_perfect_molecules(self, mask, bond_threshold):
        """
        Perfect molecules: percentage of molecules where every bond prediction is correct.
        """
        correct = (
            ((self.pair_data["bond_existence"] == 1) & (self.pair_data["predicted_bond_existence"] > bond_threshold)) |
            ((self.pair_data["bond_existence"] == 0) & (self.pair_data["predicted_bond_existence"] < bond_threshold))
        )

        return correct[mask].groupby(self.pair_data.loc[mask, "molecule_name"]).all().mean()

    def calculate_smiles_accuracy(self, bond_threshold):  # Needs tests
        results = []

        pair_groups = self.pair_data.groupby("molecule_name")
        atom_groups = self.atom_data.groupby("molecule_name")

        for mol_name in pair_groups.groups.keys():
            pair_group = pair_groups.get_group(mol_name)
            atom_group = atom_groups.get_group(mol_name)

            truth_bo = Fn.build_mol_bond_order(atom_group, pair_group, pair_group["bond_order"], rdmol=False)
            pred     = Fn.build_mol_bond_order(atom_group, pair_group, (pair_group["predicted_bond_existence"] > bond_threshold).astype(int), rdmol=False)

            results.append(pred == truth_bo)

        return np.mean(results)
  
    def SavePlots(self) :
        for plot_name, hist in self.Plots.items():
                if len(hist) == 3:  # 2D histogram (counts, x_edges, y_edges)
                    print(len(hist[0]), len(hist[1]))
                    U.plot_histogram2d(hist, title=self.Title + "_" + plot_name)
                else:               # 1D histogram
                    U.plot_histogram(data=[hist], title=self.Title + "_" + plot_name, xlabel=plot_name)

    def PrintMetrics(self, print_header=True):
        if print_header:
            print(f"{'Configuration':<30}" + "".join(f" {k:>10}" for k in self.Metrics))
            print("-" * (30 + 11 * len(self.Metrics)))
        print(f"{self.Title:<30}" + "".join(f" {v:10.4f}" for v in self.Metrics.values()))