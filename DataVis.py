import sys 
import glob
import os 
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Utils as U
from ImpDataset import ImpDataset

def main():
    
    Results_path = "/scratch/b5ao/pbellos.b5ao/Results/"

    # Bdf=[]
    # Bdf.append(ImpDataset("Original dataset_B", "Test", Results_path+"DefaultNMR_Bbond_existence_test_pairdf.parquet"))
    # Bdf.append(ImpDataset("Z Orig dataset_B", "Test", Results_path+"ZheqiOrig_test_pairdf.parquet"))
    
    # parameters = ["CC", 0.5]  #bond type, decision threshold
    # metrics    = ["Precision", "Recall", "PerfectMolecules"]
    # plots      = ["PBP", "NBP"]

    # for i, d in enumerate(Bdf):
    #     d.AnalyzeBonds(parameters, metrics, plots, 0, print_header=(i == 0))

    # U.plot_histogram([Bdf[0].Plots["PBP"],Bdf[0].Plots["NBP"], Bdf[1].Plots["PBP"], Bdf[1].Plots["NBP"]],
    #                   colors=['black','red','black','red'], linestyles=['-','-',':',':'], labels=["Pan +", "Pan -", "Zhe + ", "Zhe -"],
    #                   bins=40, x_range=[0, 1], y_range=[1, 10**8], title='TestPvsZ', xlabel="Probability", ylabel="Frequency", logy=True)

    ##========================================================================================================================================
    
    Ddf=[]
    Ddf.append(ImpDataset("Distance_Default_Train", "Train", Results_path+"DisDefNMR_Bdistance_train_pairdf.parquet"))
    Ddf.append(ImpDataset("Distance_Default_Test", "Test", Results_path+"DisDefNMR_Bdistance_test_pairdf.parquet"))

    parameters = [4]  #max path lenght
    metrics    = ["DistanceMeanError"]
    plots      = ["DistanceMEPerMolecule", "DistancePredVsTrue"]

    for i, d in enumerate(Ddf):
        d.AnalyzeDistances(parameters, metrics, plots, 0, print_header=(i == 0)) 

    U.plot_histogram([Ddf[0].Plots["DistanceMEPerMolecule"],Ddf[1].Plots["DistanceMEPerMolecule"]],
                      colors=['black','red'], linestyles=['-','-'], labels=["Train", "Test"],
                      bins=40, x_range=[-3, 3], y_range=None, title='DisTrvsTs', xlabel="Mean Error per Molecule [10$^{-10}$ m]", ylabel="Frequency", logy=False)

    U.plot_histogram2d(Ddf[0].Plots["DistancePredVsTrue"], bins=[40, 40], x_range=[0,4], y_range=[0,4], title="Dis", xlabel="True", ylabel="Pred")

if __name__ == "__main__":
    main()