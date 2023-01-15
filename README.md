![Python Version: 3.8.5](https://img.shields.io/badge/Python%20Version-3.8.5-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen)](https://opensource.org/licenses/MIT)

# Nucleotide Augmentation For Machine Learning-Guided Protein Engineering




This repository contains the scripts to perform **N**ucleo**t**ide **A**ugmentation (**NTA**) on a labeled data set of amino acid sequences as well as to recreate the analysis described in Minot & Reddy 2022 [[1](https://doi.org/10.1093/bioadv/vbac094)].


## Table of contents
1. [Prepare Working Environment](#prepare-working-environment)
2. [Apply NTA to Your Data Set](#apply-nta-to-your-data-set)
3. [Reproducing Study Results](#reproducing-study-results)
4. [Citation](#citation)


## Prepare Working Environment

#### Setup with Conda
Before running any of the scripts, the necessary packages can be installed via Conda, the open source package management system [[2](https://docs.conda.io/)], and one of the two provided environment files `nta_environment.yml` using following commands:

```console
cd nta_environment
conda env create -f nta_environment.yml
conda activate nta_env
```
#### Setup with venv
If virtualenv is preferred the environment can be setup from `nta_environment/` from `requirements.txt`, via The following 3 commands using python 3.8.5 and pip 20.1.1:
1. `python -m venv nta_env`
2. Next, on Windows, run:
`nta_env\Scripts\activate.bat`
Or, on Unix / MacOS, run:
`source nta_env/bin/activate`
3. Then pip install the requirements via: `pip install -r requirements.txt`


## Apply NTA to Your Data Set

The folder `run_nta` contains instructions and code to apply NTA to your own data. 



## Reproducing Study Results

The full pipeline to reproduce the study, written in Python, can be summarised into three consecutive steps:

 1. Data preprocessing and Nucleotide Augmentation (NTA).
 2. Model training and evaluation.
 3. Plot results.

#### Step 1 - Preprocessing and Augmentation
Unzip `data/`. Perform preprocessing by running the following commands from the main folder:

```console
cd preprocessing
./preprocessing.sh
```
This will execute train/val/test splitting and NTA for the GB1, AAV, and Trastuzumab data sets and save the resulting subsets in separate .CSV files in their respective subfolders under `data/`.


#### Step 2 - Model Training and Evaluation
Performed for each data set by executing the following commands from the main folder:
```console
cd scripts
./train_eval_gb1.sh
./train_eval_aav.sh
./train_eval_trastuzumab.sh
```

This will populate the folder `results/` with .CSV files containing the training and evaluation results for each data set and in the appropriate format for plotting in Step 3.


#### Setp 3 - Plot Results
Performed by running the following commands from the main folder:
```console
cd plot
python plot_gb1.py
python plot_aav.py
python plot_trastuzumab.py
```

## Citation

If you use the code in this repository for your research, please cite our paper.

```
@article{10.1093/bioadv/vbac094,
    author = {Minot, Mason and Reddy, Sai T},
    title = "{Nucelotide augmentation for machine learning-guided protein engineering}",
    journal = {Bioinformatics Advances},
    year = {2022},
    month = {12},
    issn = {2635-0041},
    doi = {10.1093/bioadv/vbac094},
    url = {https://doi.org/10.1093/bioadv/vbac094},
    note = {vbac094},
    eprint = {https://academic.oup.com/bioinformaticsadvances/advance-article-pdf/doi/10.1093/bioadv/vbac094/47762525/vbac094.pdf},
}
```
