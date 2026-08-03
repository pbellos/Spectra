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

    # atomdf_test_path  = "/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/FCatomdf_Test.parquet"
    # pairdf_test_path  = "/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/FCpairdf_Test.parquet"
    # atomdf, pairdf = U.MakeDataSet2(atomdf_test_path,pairdf_test_path,"NMR_B",False)
    # U.RunPrediction("/home/b5ao/pbellos.b5ao/Spectra/1shot/oneshot_B.torch" , atomdf, pairdf, None, Results_path+"1Shot_test_")
 
   
    Bdf=[]
    # Bdf.append(ImpDataset("Z-1shot", "Test", Results_path+"1Shot_test_pairdf.parquet"))
    # Bdf.append(ImpDataset("Single", "Test", Results_path+"DefaultNMR_Bbond_existence_test_pairdf.parquet"))
    # Bdf.append(ImpDataset("Parallel", "Test", Results_path+"ParTrain1.1NMR_Bbond_existence_test_pairdf.parquet"))
    # Bdf.append(ImpDataset("Parallel", "Test", Results_path+"ParTrain1.4NMR_Bbond_existence_test_pairdf.parquet"))

    parameters = ["", 0.5]  #bond type, decision threshold
    metrics    = ["TruePositiveRate", "TrueNegativeRate"]  #["Precision", "Recall", "PerfectMolecules",
    plots      = ["PBP", "NBP"]

    # for i, d in enumerate(Bdf):
    #     d.AnalyzeBonds(parameters, metrics, plots, 0, print_header=(i == 0))

    # U.plot_histogram([Bdf[0].Plots["PBP"],Bdf[0].Plots["NBP"]],
    #                    colors=['black','red'], linestyles=['-','-'], labels=["+", "-"],
    #                    bins=40, x_range=[0, 1], y_range=[1, 10**8], title='1Shot', xlabel="Probability", ylabel="Frequency", logy=True)

    df0 = pd.read_csv("/scratch/b5ao/pbellos.b5ao/Results/DefaultNMR_Bbond_existence/loss_metrics/bond_existence.csv")
    df1 = pd.read_csv("/scratch/b5ao/pbellos.b5ao/Results/ParTrainFinal1_240NMR_Bbond_existence/loss_metrics/bond_existence.csv")
    df2 = pd.read_csv("/scratch/b5ao/pbellos.b5ao/Results/ParTrainFinal2_240NMR_Bbond_existence/loss_metrics/bond_existence.csv")
    df3 = pd.read_csv("/scratch/b5ao/pbellos.b5ao/Results/ParTrain_test_4NMR_Bbond_existence/loss_metrics/bond_existence.csv")

    df40 = pd.read_csv("/scratch/b5ao/pbellos.b5ao/Results/DefaultNMR_Bbond_existence/loss_metrics/training_time.csv")
    df4 = pd.read_csv("/scratch/b5ao/pbellos.b5ao/Results/ParTrainFinal1_240NMR_Bbond_existence/loss_metrics/training_time.csv")
    df5 = pd.read_csv("/scratch/b5ao/pbellos.b5ao/Results/ParTrainFinal2_240NMR_Bbond_existence/loss_metrics/training_time.csv")
    df6 = pd.read_csv("/scratch/b5ao/pbellos.b5ao/Results/ParTrain_test_4NMR_Bbond_existence/loss_metrics/training_time.csv")

    df0 = df0.iloc[:-20]
    df40 = df40.iloc[:-20]
    df1 = df1.iloc[:-1]
    df2 = df2.iloc[:-1]
    df3 = df3.iloc[:-1]
    df_cum40 = df40.cumsum()
    df_cum4 = df4.cumsum()
    df_cum5 = df5.cumsum()
    df_cum6 = df6.cumsum()

    print(df0)
    print(df1)
    print(df2)
    print(df3)
    print(df_cum40)
    print(df_cum4)
    print(df_cum5)
    print(df_cum6)
    
    # df2 = pd.read_csv("/scratch/b5ao/pbellos.b5ao/Results/ParTrain_24BS_2NMR_Bbond_existence/loss_metrics/bond_existence.csv")
    # df3 = pd.read_csv("/scratch/b5ao/pbellos.b5ao/Results/ParTrain_24BS_3NMR_Bbond_existence/loss_metrics/bond_existence.csv")
    # df4 = pd.read_csv("/scratch/b5ao/pbellos.b5ao/Results/ParTrain_24BS_4NMR_Bbond_existence/loss_metrics/bond_existence.csv")

    U.plot_scatter(
        [df_cum40["training_time"], df_cum4["training_time"], df_cum5["training_time"] ],
        #[df0["epochs"], df1["epochs"], df2["epochs"]],
        [
            df0["train_ml_loss"],
            df1["train_ml_loss"],
            df2["train_ml_loss"],
            #df3["train_ml_loss"],
        ],
        colors=["black", "blue", "orange"],
        labels=["Old Main", "1 GPU", "2 GPUs"],
        alpha=0.7,
        s=10,
        title="LOSSvsTIME_240",
        xlabel="Epochs",
        ylabel="Loss",
    )
    
    # df = pd.read_csv(Results_path+"ParTrain1.1NMR_Bbond_existence/loss_metrics/bond_existence.csv")
 
    # U.plot_scatter([df["epochs"], df["epochs"]], [df["train_ml_loss"], df["eval_ml_loss"]],
    #                colors=["blue", "orange"], labels=["train", "eval"], alpha=0.7, s=10,
    #                title="1GPULoss", xlabel="Epoch", ylabel="Loss")

    
    # df = pd.read_csv(Results_path+"ParTrain1.4NMR_Bbond_existence/loss_metrics/bond_existence.csv")
 
    # U.plot_scatter([df["epochs"], df["epochs"]], [df["train_ml_loss"], df["eval_ml_loss"]],
    #                colors=["blue", "orange"], labels=["train", "eval"], alpha=0.7, s=10,
    #                title="4GPULoss", xlabel="Epoch", ylabel="Loss")

    # df = pd.read_csv(Results_path+"ParTrain2.4NMR_Bbond_existence/loss_metrics/bond_existence.csv")
 
    # U.plot_scatter([df["epochs"], df["epochs"]], [df["train_ml_loss"], df["eval_ml_loss"]],
    #                colors=["blue", "orange"], labels=["train", "eval"], alpha=0.7, s=10,
    #                title="24GPULoss", xlabel="Epoch", ylabel="Loss")

    
    # df = pd.read_csv(Results_path+"ParTrain1.4_FBSNMR_Bbond_existence/loss_metrics/bond_existence.csv")
 
    # U.plot_scatter([df["epochs"], df["epochs"]], [df["train_ml_loss"], df["eval_ml_loss"]],
    #                colors=["blue", "orange"], labels=["train", "eval"], alpha=0.7, s=10,
    #                title="FB_4GPULoss", xlabel="Epoch", ylabel="Loss")

    
    # df = pd.read_csv(Results_path+"ParTrain1.4_2lr_4BSNMR_Bbond_existence/loss_metrics/bond_existence.csv")
 
    # U.plot_scatter([df["epochs"], df["epochs"]], [df["train_ml_loss"], df["eval_ml_loss"]],
    #                colors=["blue", "orange"], labels=["train", "eval"], alpha=0.7, s=10,
    #                title="2lr_FB_4GPULoss", xlabel="Epoch", ylabel="Loss")

    # df = pd.read_csv(Results_path+"ParTrain1.4_DSNMR_Bbond_existence/loss_metrics/bond_existence.csv")
 
    # U.plot_scatter([df["epochs"], df["epochs"]], [df["train_ml_loss"], df["eval_ml_loss"]],
    #                colors=["blue", "orange"], labels=["train", "eval"], alpha=0.7, s=10,
    #                title="DS_4GPULoss", xlabel="Epoch", ylabel="Loss")


    # df = pd.read_csv(Results_path+"ParTrain1.4_F_16BS_BigLRNMR_Bbond_existence/loss_metrics/bond_existence.csv")
 
    # U.plot_scatter([df["epochs"], df["epochs"]], [df["train_ml_loss"], df["eval_ml_loss"]],
    #                colors=["blue", "orange"], labels=["train", "eval"], alpha=0.7, s=10,
    #                title="BLR_4GPULoss", xlabel="Epoch", ylabel="Loss")




  

    ##========================================================================================================================================
    
    # Ddf=[]
    # Ddf.append(ImpDataset("Distance_Default_Train", "Train", Results_path+"DisDefNMR_Bdistance_train_pairdf.parquet"))
    # Ddf.append(ImpDataset("Distance_Default_Test", "Test", Results_path+"DisDefNMR_Bdistance_test_pairdf.parquet"))

    # parameters = [40]  #max path lenght
    # metrics    = ["DistanceMeanError"]
    # plots      = ["DistanceMEPerMolecule", "DistancePredVsTrue"]

    # for i, d in enumerate(Ddf):
    #     d.AnalyzeDistances(parameters, metrics, plots, 0, print_header=(i == 0)) 

    # U.plot_histogram([Ddf[0].Plots["DistanceMEPerMolecule"],Ddf[1].Plots["DistanceMEPerMolecule"]],
    #                   colors=['black','red'], linestyles=['-','-'], labels=["Train", "Test"],
    #                   bins=40, x_range=[-3, 3], y_range=None, title='DisTrvsTs', xlabel="Mean Error per Molecule [10$^{-10}$ m]", ylabel="Frequency", logy=False)

    # U.plot_histogram2d(Ddf[0].Plots["DistancePredVsTrue"], bins=[40, 40], x_range=[0,10], y_range=[0,10], title="Dis", xlabel="True", ylabel="Pred")

if __name__ == "__main__":
    main()