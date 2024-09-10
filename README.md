# ESM_MMP_WTMP
## Protein Mutation Prediction Script

This repository contains a Python script for predicting the effects of mutations on a protein sequence using the ESM (Evolutionary Scale Modeling) model. The script generates heatmaps that display the impact of mutations at each position of a protein sequence. Two types of predictions are provided:

	•	MMP (Masked Model Prediction)
	•	WTMP (Wildtype Model Prediction)

## Prerequisites

To run this script, you will need the following dependencies:

	•	Python 3.x
	•	torch (PyTorch)
	•	esm (ESM model from Facebook AI Research)
	•	numpy
	•	argparse
	•	matplotlib
	•	tqdm

You can install the dependencies via pip:
```bash
pip install torch esm numpy matplotlib tqdm
```
## Usage

The script can be executed by providing input parameters, such as the path to the input file and output directory, through the command line.

## Command Line Arguments:

	•	-i : Path to the input file containing the designed protein sequence in FASTA format.
	•	-o : Directory where the output files (heatmaps and results) will be saved.

 ## Example:
```bash
python casl_esm_score_matrix.py -i 5zkn.fa -o ./
```


## Input File Format

The input file should contain the designed protein sequence in plain text format\(fasta\).

## Output Files

The script generates the following outputs:

* __MMP_result.fasta__: Contains the MMP scores and the corresponding mutated sequences in FASTA format.
* __WTMP_result.fasta__: Contains the WTMP scores and the corresponding mutated sequences in FASTA format.
* __MMP_heatmap.pdf__: A heatmap showing the predicted effects of mutations on the protein sequence based on the MMP model.
* __WTMP_heatmap.pdf__: A heatmap showing the predicted effects of mutations on the protein sequence based on the WTMP model.

## How It Works

* __Load the ESM Model__: The script loads the ESM2 model (esm2_t33_650M_UR50D) to make predictions about the protein sequence.
* __Mutation Simulation__: For each position in the input sequence, the script simulates mutations to all possible amino acids, calculates the MMP and WTMP scores using the loaded model, and records the results.
* __Heatmap Generation__: The calculated mutation scores are used to create two heatmaps (MMP and WTMP), showing the likelihood of mutations at different positions along the protein sequence.
* __Save Results__: The scores and mutated sequences are saved in FASTA format, and the heatmaps are saved as PDFs.


