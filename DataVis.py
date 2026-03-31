import sys 
import glob
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Functions as Fn

class ImpDataset:

    def __init__(self, Title="Test", path="Test1NMR_Zbond_existence_pairdf", color='black', line_style='-'):
        self.Title = Title
        self.path = path
        self.color = color
        self.line_style = line_style
        self.data = pd.read_parquet(path+"pairdf.parquet") 
        print(self.data.head(1000).to_string())
        print(f"Loaded {self.Title} with {len(self.data)} rows")

    def BondMetrics(self, bond_filter="" , bond_threshold=0.5, print_header=True) :

        mask = self.data["nmr_types"].str.contains(bond_filter, na=False)

        pos_mask = (self.data["bond_existence"] == 1) & mask
        neg_mask = (self.data["bond_existence"] == 0) & mask

        pos_per_bond = ((pos_mask & (self.data["predicted_bond_existence"] > bond_threshold)).sum()/ pos_mask.sum())
        neg_per_bond = ((neg_mask & (self.data["predicted_bond_existence"] < bond_threshold)).sum()/ neg_mask.sum())

        recall = ((pos_mask & (self.data["predicted_bond_existence"] > bond_threshold)).sum())/(((pos_mask & (self.data["predicted_bond_existence"] > bond_threshold)).sum())+((neg_mask & (self.data["predicted_bond_existence"] > bond_threshold)).sum()))

        correct = (((self.data["bond_existence"] == 1)  & (self.data["predicted_bond_existence"] > bond_threshold)) | ((self.data["bond_existence"] == 0) & (self.data["predicted_bond_existence"] < bond_threshold)))

        correct_molecules = (correct[mask].groupby(self.data.loc[mask, "molecule_name"]).all().mean())
        
        if print_header==True : 
            print(f"{'Configuration':<30} {'precision':>10} {'recall':>10} {'100% mol':>10}")
            print("-"*60)
        print(f"{self.Title:<30} " f"{pos_per_bond:10.2%} " f"{recall:10.2%} " f"{correct_molecules:10.2%}")


def main():
    
    df=[]

    atomdf_test_path  = "/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/atomdf_Test.parquet"
    pairdf_test_path  = "/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/pairdf_Test.parquet"
    Results_path = "/scratch/b5ao/pbellos.b5ao/Results/"

    # atomdf_test,pairdf_test = Fn.MakeDataSet2(atomdf_test_path,pairdf_test_path,"NMR_B",True)

    # Fn.RunPrediction("/projects/b5ao/public/zheqi_model/oneshot_new_embedding.torch"      , atomdf_test, pairdf_test, None, Results_path+"ZheqiNew2_test_")
    # Fn.RunPrediction("/projects/b5ao/public/zheqi_model/oneshot_original_embedding.torch" , atomdf_test, pairdf_test, None, Results_path+"ZheqiOrig2_test_")
    
    df.append(ImpDataset("ZNew B OPT Test", Results_path+"ZheqiNew2_test_", 'blue',':'))
    df.append(ImpDataset("ZOrig B OPT Test", Results_path+"ZheqiOrig2_test_", 'blue',':'))
    #df.append(ImpDataset("Default_Imp B OPT Test", Results_path+"DefaultNMR_Bbond_existence_test_", 'green','--'))

    # df.append(ImpDataset("EMB2_ws10NMR_A_Imp B OPT Test", Results_path+"EMB2_ws10NMR_Abond_existence_test_", 'blue','-'))
    # df.append(ImpDataset("EMB2_ws10NMR_B_Imp A OPT Test", Results_path+"EMB2_ws10NMR_Bbond_existence_test_", 'blue','-'))
     
    bond_thresholds = [0.5] #0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95]
    bond_types = ["", "CC", "CH", "OC", "NH", "FC"]
    
    for bt in bond_thresholds : 
      for i, d in enumerate(df):
          d.BondMetrics(bond_filter='', bond_threshold=bt, print_header=(i == 0))

    Fn.plot_histogram([df[0].data.loc[(df[0].data["bond_existence"] == 1) & df[0].data["nmr_types"].str.contains(bond_types[0], na=False),"predicted_bond_existence"], 
                       df[1].data.loc[(df[1].data["bond_existence"] == 0) & df[1].data["nmr_types"].str.contains(bond_types[0], na=False),"predicted_bond_existence"]], 
                       bins=50, x_range=[0, 1], y_range=[1, 10**8], title='TestPlot', xlabel="Probability", ylabel="Frequency", color=['blue','red'], linestyle=d.line_style, logy=True, newFigure=True)

      
if __name__ == "__main__":
    main()




    # df.append(ImpDataset("Default_Imp A OPT Train", "DefaultNMR_Abond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("Default_Imp A OPT Test", "DefaultNMR_Abond_existence_pairdf_test", 'blue',':'))
    # df.append(ImpDataset("Default_Imp B OPT Train", "DefaultNMR_Bbond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("Default_Imp B OPT Test", "DefaultNMR_Bbond_existence_pairdf_test", 'blue',':'))


   #df.append(ImpDataset("Default_Imp Z", "Test1NMR_Zbond_existence_pairdf", 'black'))
    #df.append(ImpDataset("Default_Imp A 25 Train", "DefaultNMR_Abond_existence_pairdf_train25", 'red','-'))
    #df.append(ImpDataset("Default_Imp A 25 Eval", "DefaultNMR_Abond_existence_pairdf_eval25", 'green','--'))
    #df.append(ImpDataset("Default_Imp A 25 Test", "DefaultNMR_Abond_existence_pairdf_test25", 'blue',':'))
    
    # df.append(ImpDataset("Default_Imp A OPT Eval", "DefaultNMR_Abond_existence_pairdf_eval", 'green','--'))
    

    # df.append(ImpDataset("EMB2_Imp A 25 Train", "EMB2NMR_Abond_existence_pairdf_train25", 'red','-'))
    # df.append(ImpDataset("EMB2_Imp A 25 Eval", "EMB2NMR_Abond_existence_pairdf_eval25", 'green','--'))
    # df.append(ImpDataset("EMB2_Imp A 25 Test", "EMB2NMR_Abond_existence_pairdf_test25", 'blue',':'))
    # df.append(ImpDataset("EMB2_Imp A OPT Train", "EMB2NMR_Abond_existence_pairdf_train", 'red','-'))
    # # # df.append(ImpDataset("EMB2_Imp A OPT Eval", "EMB2NMR_Abond_existence_pairdf_eval", 'green','--'))
    #df.append(ImpDataset("EMB2_Imp A OPT Test", "EMB2NMR_Abond_existence_pairdf_test", 'blue',':'))

    #df.append(ImpDataset("EMB2_Imp B OPT Train", "EMB2NMR_Bbond_existence_pairdf_train", 'red','-'))
    # # # df.append(ImpDataset("EMB2_Imp B OPT Eval", "EMB2NMR_Bbond_existence_pairdf_eval", 'green','--'))
    #df.append(ImpDataset("EMB2_Imp B OPT Test", "EMB2NMR_Bbond_existence_pairdf_test", 'blue',':'))

    
    # df.append(ImpDataset("EMB2_sm_Imp A OPT Train", "EMB2_smNMR_Abond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_sm_Imp A OPT Eval", "EMB2_smNMR_Abond_existence_pairdf_eval", 'green','--'))
    # df.append(ImpDataset("EMB2_sm_Imp A OPT Test", "EMB2_smNMR_Abond_existence_pairdf_test", 'blue',':'))

    # df.append(ImpDataset("Default_Imp B 25 Train", "DefaultNMR_Bbond_existence_pairdf_train25", 'red','-'))
    # df.append(ImpDataset("Default_Imp B 25 Eval", "DefaultNMR_Bbond_existence_pairdf_eval25", 'green','--'))
    # df.append(ImpDataset("Default_Imp B 25 Test", "DefaultNMR_Bbond_existence_pairdf_test25", 'blue',':'))

    # df.append(ImpDataset("Default_Imp B OPT Eval", "DefaultNMR_Bbond_existence_pairdf_eval", 'green','--'))



    # df.append(ImpDataset("EMB2_dim35_Imp A OPT Train", "EMB2_dim35NMR_Abond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_dim35_Imp A OPT Test", "EMB2_dim35NMR_Abond_existence_pairdf_test", 'blue',':'))
    # df.append(ImpDataset("EMB2_dim35_Imp B OPT Train", "EMB2_dim35NMR_Bbond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_dim35_Imp B OPT Test", "EMB2_dim35NMR_Bbond_existence_pairdf_test", 'blue',':'))

    # df.append(ImpDataset("EMB2_dim60_Imp A OPT Train", "EMB2_dim60NMR_Abond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_dim60_Imp A OPT Test", "EMB2_dim60NMR_Abond_existence_pairdf_test", 'blue',':'))
    # df.append(ImpDataset("EMB2_dim60_Imp B OPT Train", "EMB2_dim60NMR_Bbond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_dim60_Imp B OPT Test", "EMB2_dim60NMR_Bbond_existence_pairdf_test", 'blue',':'))

    # df.append(ImpDataset("EMB2_L5_Imp A OPT Train", "EMB2_L5NMR_Abond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_L5_Imp A OPT Test", "EMB2_L5NMR_Abond_existence_pairdf_test", 'blue',':'))
    # df.append(ImpDataset("EMB2_L5_Imp B OPT Train", "EMB2_L5NMR_Bbond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_L5_Imp B OPT Test", "EMB2_L5NMR_Bbond_existence_pairdf_test", 'blue',':'))

    # df.append(ImpDataset("EMB2_L7_Imp A OPT Train", "EMB2_L7NMR_Abond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_L7_Imp A OPT Test", "EMB2_L7NMR_Abond_existence_pairdf_test", 'blue',':'))
    # df.append(ImpDataset("EMB2_L7_Imp B OPT Train", "EMB2_L7NMR_Bbond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_L7_Imp B OPT Test", "EMB2_L7NMR_Bbond_existence_pairdf_test", 'blue',':'))


    # df.append(ImpDataset("EMB2_noNMRNMR_A_Imp A OPT Train", "EMB2_noNMRNMR_Abond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_noNMRNMR_A_Imp A OPT Test", "EMB2_noNMRNMR_Abond_existence_pairdf_test", 'blue',':'))
    # df.append(ImpDataset("EMB2_noNMRNMR_B_Imp B OPT Train", "EMB2_noNMRNMR_Bbond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_noNMRNMR_B_Imp B OPT Test", "EMB2_noNMRNMR_Bbond_existence_pairdf_test", 'blue',':'))

    # df.append(ImpDataset("EMB2_noNMR_L5NMR_A_Imp A OPT Train", "EMB2_noNMR_L5NMR_Abond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_noNMR_L5NMR_A_Imp A OPT Test", "EMB2_noNMR_L5NMR_Abond_existence_pairdf_test", 'blue',':'))
    # df.append(ImpDataset("EMB2_noNMR_L5NMR_B_Imp B OPT Train", "EMB2_noNMR_L5NMR_Bbond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_noNMR_L5NMR_B_Imp B OPT Test", "EMB2_noNMR_L5NMR_Bbond_existence_pairdf_test", 'blue',':'))

    # df.append(ImpDataset("EMB2_ws1NMR_A_Imp A OPT Train", "EMB2_ws1NMR_Abond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_ws1NMR_A_Imp A OPT Test", "EMB2_ws1NMR_Abond_existence_pairdf_test", 'blue',':'))
    # df.append(ImpDataset("EMB2_ws1NMR_B_Imp B OPT Train", "EMB2_ws1NMR_Bbond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_ws1NMR_B_Imp B OPT Test", "EMB2_ws1NMR_Bbond_existence_pairdf_test", 'blue',':'))

    # df.append(ImpDataset("EMB2_ws10NMR_A_Imp A OPT Train", "EMB2_ws10NMR_Abond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_ws10NMR_A_Imp A OPT Test", "EMB2_ws10NMR_Abond_existence_pairdf_test", 'blue',':'))
    # df.append(ImpDataset("EMB2_ws10NMR_B_Imp B OPT Train", "EMB2_ws10NMR_Bbond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_ws10NMR_B_Imp B OPT Test", "EMB2_ws10NMR_Bbond_existence_pairdf_test", 'blue',':'))

    # df.append(ImpDataset("ZN B OPT Train", "ZN_B_pairdf_train", 'red','-'))
    # df.append(ImpDataset("ZN B OPT Test", "ZN_B_pairdf_test", 'blue',':'))
    # df.append(ImpDataset("ZO B OPT Train", "ZO_B_pairdf_train", 'red','-'))
    # df.append(ImpDataset("ZO B OPT Test", "ZO_B_pairdf_test", 'blue',':'))


