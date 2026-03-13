import sys
import glob
import os
from tqdm import tqdm
import pandas as pd
from EMS.EMS import EMS
from EMS.modules.dataframe_generation.dataframe_parse import make_atoms_df, make_pairs_df
 
batches = glob.glob("/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/PubChem_CHEMBL-B1234-30-200000_*")

with open('/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/PubChem_CHEMBL-B1234-30-200000_2000_eval.sdf', 'r') as f:
    lines = f.read()
    mol_block = lines.split('$$$$\n')
    mol_blocks = [block for block in mol_block if block.strip()]
 
ems_list = []

I=0
 
for block in mol_blocks:

    I=I+1

    with open('tmp.sdf', 'w') as f:
        f.write(block)
    try:
        ems_instance = EMS('tmp.sdf', nmr=True)
    except:
        continue
 
    ems_list.append(ems_instance)

    # if I>100 :
    #     break

print(f"Number of valid molecules: {len(ems_list)}")


atomdf = make_atoms_df(ems_list)
pairdf = make_pairs_df(ems_list)


atomdf.to_parquet(f"/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/atomdf_Test.parquet")
pairdf.to_parquet(f"/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/pairdf_Test.parquet")














""" 
emols = []
emol = EMS(file="/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/PubChem_CHEMBL-B1234-30-200000_10000_train.sdf", nmr=True)
emol.get_coupling_types()
emols.append(emol)

atomdf = make_atoms_df(emols)
pairdf = make_pairs_df(emols)
 
atomdf.to_parquet(f"/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/atomdf_Test.parquet")
pairdf.to_parquet(f"/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/pairdf_Test.parquet")


 """

""" for batch in enumerate(batches):
    batch_name = os.path.basename(batch)
    print(batch_name)

    emols = []
    emol = EMS(file=batch_name, nmr=True)

    atomdf = make_atoms_df(emols)
    pairdf = make_pairs_df(emols)
 
    atomdf.to_parquet(f"/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/atomdf_{batch_name}.parquet")
    pairdf.to_parquet(f"/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/pairdf_{batch_name}.parquet")
 """



"""     cifs = glob.glob(f"{batch}/*.cif")
    print(cifs)
 
    emols = []
    for cif in tqdm(cifs, total=len(cifs), desc=f'Batch {idx}'):
        try:
            emol = EMS(cif, mol_id = os.path.splitext(os.path.basename(cif))[0].split('.')[0])
            print (os.path.splitext(os.path.basename(cif))[0].split('.')[0])
            emol.get_coupling_types()
            emols.append(emol)
        except Exception as e:
            print(f"Error processing {cif}: {e}") """
 



## Print few examples 

print(atomdf['molecule_name'].unique())

print(atomdf.iloc[[0]])
print(atomdf.iloc[[1]])
print(atomdf.iloc[[2]])
print("...............")
print(atomdf.iloc[[100]])
print(atomdf.iloc[[101]])
print(atomdf.iloc[[102]])

print("===========")
print(pairdf.iloc[[0]])
print(pairdf.iloc[[1]])
print(pairdf.iloc[[2]])
print("...............")
print(pairdf.iloc[[1000]])
print(pairdf.iloc[[1001]])
print(pairdf.iloc[[1002]])