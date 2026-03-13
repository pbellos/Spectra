import sys 
import glob
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

class ImpDataset:

    def __init__(self, Title="Test", path="Test1NMR_Zbond_existence_pairdf", color='black', line_style='-'):
        self.Title = Title
        self.path = path
        self.color = color
        self.line_style = line_style
        self.data = pd.read_parquet("/home/b5ao/pbellos.b5ao/Spectra/Results/"+path+".parquet") 

    def BondMetrics(self, print_header=True) :

        #print(self.data.head(50))
        pos_per_bond = ((self.data["bond_existence"] == 1) & (self.data["predicted_bond_existence"] > 0.5)).sum()/(self.data["bond_existence"] == 1).sum()
        neg_per_bond = ((self.data["bond_existence"] == 0) & (self.data["predicted_bond_existence"] < 0.5)).sum()/(self.data["bond_existence"] == 0).sum()
        correct = (((self.data["bond_existence"] == 1) & (self.data["predicted_bond_existence"] > 0.5)) | ((self.data["bond_existence"] == 0) & (self.data["predicted_bond_existence"] < 0.5)))
        correct_molecules = correct.groupby(self.data["molecule_name"]).all().mean()
        
        if print_header==True : 
            print(f"{'Configuration':<25} {'pos/bond':>10} {'neg/bond':>10} {'per mol':>10}")
            print("-"*60)
        print(f"{self.Title:<25} " f"{pos_per_bond:10.3%} " f"{neg_per_bond:10.3%} " f"{correct_molecules:10.3%}")





def main():
    
    df=[]
    #df.append(ImpDataset("Default_Imp Z", "Test1NMR_Zbond_existence_pairdf", 'black'))
    # df.append(ImpDataset("Default_Imp A 25 Train", "DefaultNMR_Abond_existence_pairdf_train25", 'red','-'))
    # df.append(ImpDataset("Default_Imp A 25 Eval", "DefaultNMR_Abond_existence_pairdf_eval25", 'green','--'))
    # df.append(ImpDataset("Default_Imp A 25 Test", "DefaultNMR_Abond_existence_pairdf_test25", 'blue',':'))
    # df.append(ImpDataset("Default_Imp A OPT Train", "DefaultNMR_Abond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("Default_Imp A OPT Eval", "DefaultNMR_Abond_existence_pairdf_eval", 'green','--'))
    # df.append(ImpDataset("Default_Imp A OPT Test", "DefaultNMR_Abond_existence_pairdf_test", 'blue',':'))

    # df.append(ImpDataset("EMB2_Imp A 25 Train", "EMB2NMR_Abond_existence_pairdf_train25", 'red','-'))
    # df.append(ImpDataset("EMB2_Imp A 25 Eval", "EMB2NMR_Abond_existence_pairdf_eval25", 'green','--'))
    # df.append(ImpDataset("EMB2_Imp A 25 Test", "EMB2NMR_Abond_existence_pairdf_test25", 'blue',':'))
    # df.append(ImpDataset("EMB2_Imp A OPT Train", "EMB2NMR_Abond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_Imp A OPT Eval", "EMB2NMR_Abond_existence_pairdf_eval", 'green','--'))
    # df.append(ImpDataset("EMB2_Imp A OPT Test", "EMB2NMR_Abond_existence_pairdf_test", 'blue',':'))

    df.append(ImpDataset("Default_Imp B 25 Train", "DefaultNMR_Bbond_existence_pairdf_train25", 'red','-'))
    df.append(ImpDataset("Default_Imp B 25 Eval", "DefaultNMR_Bbond_existence_pairdf_eval25", 'green','--'))
    df.append(ImpDataset("Default_Imp B 25 Test", "DefaultNMR_Bbond_existence_pairdf_test25", 'blue',':'))
    df.append(ImpDataset("Default_Imp B OPT Train", "DefaultNMR_Bbond_existence_pairdf_train", 'red','-'))
    df.append(ImpDataset("Default_Imp B OPT Eval", "DefaultNMR_Bbond_existence_pairdf_eval", 'green','--'))
    df.append(ImpDataset("Default_Imp B OPT Test", "DefaultNMR_Bbond_existence_pairdf_test", 'blue',':'))

    # df.append(ImpDataset("EMB2_Imp B 25 Train", "EMB2NMR_Bbond_existence_pairdf_train25", 'red','-'))
    # df.append(ImpDataset("EMB2_Imp B 25 Eval", "EMB2NMR_Bbond_existence_pairdf_eval25", 'green','--'))
    # df.append(ImpDataset("EMB2_Imp B 25 Test", "EMB2NMR_Bbond_existence_pairdf_test25", 'blue',':'))
    # df.append(ImpDataset("EMB2_Imp B OPT Train", "EMB2NMR_Bbond_existence_pairdf_train", 'red','-'))
    # df.append(ImpDataset("EMB2_Imp B OPT Eval", "EMB2NMR_Bbond_existence_pairdf_eval", 'green','--'))
    # df.append(ImpDataset("EMB2_Imp B OPT Test", "EMB2NMR_Bbond_existence_pairdf_test", 'blue',':'))

    for i, d in enumerate(df):
        d.BondMetrics(print_header=(i == 0))

    
    # plt.figure()
    # for d in df :
    #   plot_histogram(d.data["bond_existence"], bins=100, x_range=[0,1], y_range=[1, 10**8], title=d.Title, xlabel="Probability", ylabel="Frequency", color=d.color, linestyle='-', logy=True, newFigure=False)
    
    # plt.legend()
    # plt.savefig(f"Plots/BE_test.png")
    # plt.close()
        
    
    #print(pairdf.columns)
    #print(pairdf.head)
    # print(len(atomdf["molecule_name"].unique()))

    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)


    #pairdf = pd.read_parquet("/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/pairdf_Train.parquet")
    #pairdf = pd.read_parquet("/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/pairdf_Train.parquet")

    # print("made pairdf")


    # plot_histogram(atomdf["typeint"], bins=10, x_range=(0,10), title="AtomType", xlabel="Atom type")
    # plot_scatter(pairdf["coupling"], pairdf["distance"], "DisVsCoup", xlabel="Coupling (?)", ylabel="Distance ($10^{-10}$ m)", alpha=0.7, s=10)

    #plot_histogram2d(pairdf["distance"], pairdf["predicted_distance"], binsx=50, binsy=50, title="DisVsPrDis_2", xlabel="Distance ($10^{-10}$ m)", ylabel="Predicted Distance ($10^{-10}$ m)", x_range=(0,10), y_range=(0,10))
    #plot_histogram2d(pairdf.loc[pairdf["bond_existence"]==1,"distance"], pairdf.loc[pairdf["bond_existence"]==1,"predicted_distance"], binsx=20, binsy=50, title="DisVsPrDis_1bond", xlabel="Distance ($10^{-10}$ m)", ylabel="Predicted Distance ($10^{-10}$ m)", x_range=(0,2), y_range=(0,5))

if __name__ == "__main__":
    main()
