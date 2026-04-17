import sys 
import glob
import os 
import math
import pandas as pd
import numpy as np
from rdkit import Chem
import subprocess
from multiprocessing import Pool
import re

def safe_name(name):
    match = re.search(r'(CHEMBL\d+)', name)
    if match:
        return match.group(1)
    else:
        # fallback: fully sanitized name if CHEMBL not found
        return re.sub(r'[^A-Za-z0-9._-]', '_', name)

# ==============================
# CONFIG
# ==============================
ORCA_CMD = "orca"
N_CORES = 8
WORKDIR = "orca_jobs"
os.makedirs(WORKDIR, exist_ok=True)

# IR grid
GRID = np.linspace(400, 4000, 2000)
SIGMA = 20
SCALE = 0.97

# ==============================
# 1) GROUP MOLECULES
# ==============================
def build_molecules(df):
    molecules = []

    for mol_id, group in df.groupby("molecule_name"):
        group = group.sort_values("atom_index")

        atoms = group["typestr"].tolist()
        coords = group[["x", "y", "z"]].values

        charge = 0
        multiplicity = 1  # assume singlet (adjust if needed)

        molecules.append({
            "mol_id": mol_id,
            "atoms": atoms,
            "coords": coords,
            "charge": charge,
            "mult": multiplicity
        })

    return molecules

# ==============================
# 2) WRITE FILES
# ==============================
def write_xyz(mol, path):
    atoms = mol["atoms"]
    coords = mol["coords"]

    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n\n")
        for a, (x,y,z) in zip(atoms, coords):
            f.write(f"{a} {x:.6f} {y:.6f} {z:.6f}\n")


def write_orca_input(mol, xyz_path, inp_path):
    with open(inp_path, "w") as f:
        #f.write("! B3LYP def2-SVP Opt Freq TightSCF\n\n")
        f.write("! B3LYP def2-SVP Freq TightSCF SlowConv\n\n")
        #f.write("! B3LYP def2-SVP Freq TightSCF\n\n")
        f.write(f"* xyzfile {mol['charge']} {mol['mult']} {xyz_path}\n")

# ==============================
# 3) RUN ORCA
# ==============================
def run_orca(inp_path):
    out_path = inp_path.replace(".inp", ".out")

    with open(out_path, "w") as f:
        result = subprocess.run([ORCA_CMD, inp_path],stdout=f,stderr=subprocess.PIPE,text=True)

    return out_path

# ==============================
# 4) PARSE IR DATA
# ==============================
def parse_ir(out_file):
    freqs = []
    intensities = []

    if not os.path.exists(out_file):
        return None, None

    with open(out_file) as f:
        lines = f.readlines()

    capture = False

    for line in lines:
        if "VIBRATIONAL FREQUENCIES" in line:
            capture = True
            continue

        if capture:
            if line.strip() == "":
                continue
            if "cm**-1" in line:
                continue

            parts = line.split()
            if len(parts) >= 3:
                try:
                    freq = float(parts[1])
                    inten = float(parts[2])

                    # skip non-physical modes
                    if freq > 50:
                        freqs.append(freq)
                        intensities.append(inten)
                except:
                    pass

    return freqs, intensities

# ==============================
# 5) BUILD SPECTRUM
# ==============================
def build_spectrum(freqs, intensities):
    if freqs is None or intensities is None:
        return None

    freqs = np.array(freqs)
    intensities = np.array(intensities)

    # stack as columns: frequency, intensity
    data = np.column_stack((freqs, intensities))

    return data

# ==============================
# 6) FULL PIPELINE
# ==============================
def process_molecule(mol):
    mol_id = mol["mol_id"]

    M=safe_name(mol_id)

    xyz_path = os.path.join(WORKDIR, f"{M}.xyz")
    inp_path = os.path.join(WORKDIR, f"{M}.inp")

    write_xyz(mol, xyz_path)
    write_orca_input(mol, xyz_path, inp_path)

    out_path = run_orca(inp_path)

    freqs, intensities = parse_ir(out_path)
    spectrum = build_spectrum(freqs, intensities)

    return M, spectrum


# ==============================
# MAIN EXECUTION
# ==============================
def run_pipeline(df):
    molecules = build_molecules(df)

    os.makedirs("spectra", exist_ok=True)

    for mol in molecules:
        mol_id, spec = process_molecule(mol)

        out_path = os.path.join("spectra", f"{mol_id}.txt")

        if spec is not None:
            np.savetxt(out_path, spec, header="frequency(cm^-1) intensity(km/mol)", fmt="%.6f")
        else:
            print(f"Skipping {mol_id}, no spectrum")
        
        break

            


if __name__ == "__main__":
    import pandas as pd

    # load your dataframe (example)
    atomdf_test_path  = "/home/b5ao/pbellos.b5ao/Spectra/Datasets/SolutionNMRraw/aTEST.parquet"
    df = pd.read_parquet(atomdf_test_path)
 
    print(df.head(100))

    df_out = run_pipeline(df)