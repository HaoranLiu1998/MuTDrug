# SupplementaryMaterial

This repository provides the supplementary materials for the Bioinformatics submission:
Multi-Target Drug Design with Bifurcated Topological Protein Representation and Multi-Modal Adaptive Spline-Attention Fusion.

## Folder Structure

```
MuTDrug_SupplementaryMaterial/
│
├── data/
│   ├── 1screenpretraindatafromZINC15.py
│   └── 2Multimodaldata.py
│
├── model/
│   ├── 1pretrainedmodel.py
│   ├── 2MuTDrug.py
│   └── 3MultiMultitarget-drugdesign.py
│
└── result/
    ├── MEK1_protein.pdb
    ├── mTOR_protein.pdb
    ├── MEK1_pocket1_site1.pdb
    ├── MEK1_pocket1_site2.pdb
    ├── ...
    ├── mTOR_pocket1_site1.pdb
    ├── mTOR_pocket1_site2.pdb
    ├── ...
    ├── MEK1_visualization.mp4
    └── mTOR_visualization.mp4

```

---

## Contents

### 📁 `data/`

This folder contains scripts for data preprocessing:

- **`1screenpretraindatafromZINC15.py`**  
  Preprocesses large-scale molecular data from the ZINC15 database for model pretraining.

- **`2Multimodaldata.py`**  
  Prepares the multimodal, multi-target dataset by integrating molecular and protein features.

---

### 📁 `model/`

This folder contains model definition and training scripts:

- **`1pretrainedmodel.py`**  
  Defines the pretraining architecture for molecule representation learning.

- **`2MuTDrug.py`**  
  Implements the complete model training pipeline for our proposed multi-target drug design framework.

- **`3MultiMultitarget-drugdesign.py`**  
  Generates multi-target molecules conditioned on multimodal target features.

---

### 📁 `result/`

This folder contains docking and visualization results:

- **PDB structures**  
  Includes the 3D structures of target proteins (e.g., MEK1, mTOR) and 9 docking conformations for each with candidate drugs.

- **Visualization videos**  
  Molecular docking visualizations for MEK1 and mTOR saved in .mp4 format.
In the videos, target proteins are displayed as surface structures, while candidate molecules are shown as stick models, providing a clearer view of the binding poses and spatial relationships.

---

## How to Use

1. **Preprocess Data**  
   Run the scripts in `data/` to generate formatted datasets for pretraining and multi-target training.

2. **Train Models**  
   - Use `1pretrainedmodel.py` for pretraining on ZINC15 data.
   - Use `2MuTDrug.py` to train the full multi-target generation model.
   - Use `3MultiMultitarget-drugdesign.py` to generate new compounds based on protein features.

3. **View Results**  
   - Examine the docking structures (`.pdb`) in molecular visualization tools (e.g., PyMOL, Chimera).
   - Play the visualization videos in any media player to see docking outcomes.

---
Citation

The citation will be released once the article is accepted.
