![Python Version: 3.8.5](https://img.shields.io/badge/Python%20Version-3.8.5-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen)](https://opensource.org/licenses/MIT)

# Nucleotide Augmentation For Machine Learning-Guided Protein Engineering




This repository contains the scripts to perform **N**ucleo**t**ide **A**ugmentation (**NTA**) on a labeled data set of amino acid sequences as well as to recreate the analysis described in Minot & Reddy 2022 [[1](https://www.biorxiv.org/content/10.1101/2022.03.08.483422v1)].


## Table of contents
1. [Prepare Working Environment](#prepare-working-environment)
2. [Apply NTA to Your Data Set](#apply-nta-to-your-data-set)
3. [Reproducing Study Results](#reproducing-study-results)
4. [Citation](#citation)


## Prepare Working Environment

2 environments are provided. The first is the full environment from the paper. The other is a minimal version which has not been extensively tested.
Certain packages in the full environment, such as apex [[3](https://nvidia.github.io/apex/)], may require manual installation per instructions provided by the authors.

Note: For the following environment activation instructions, `minimal` is used. To enable the environment used for the paper, simply replace `minimal` with `full`.

#### Setup with Conda
Before running any of the scripts, the necessary packages can be installed via Conda, the open source package management system [[2](https://docs.conda.io/)], and one of the two provided environment files `nta_environment_minimal.yml` or `nta_environment_full.yml`, using following commands:

```console
cd nta_environment
conda env create -f nta_environment_minimal.yml
conda activate nta_env_minimal
```
#### Setup with venv
If virtualenv is preferred the environment can be setup from `nta_environment/` from either `requirements_minimal.txt` or `requirements_full.txt`, via The following 3 commands using python 3.8.5 and pip 20.1.1:
1. `python -m venv nta_env_minimal`
2. Next, on Windows, run:
`nta_env_minimal\Scripts\activate.bat`
Or, on Unix / MacOS, run:
`source nta_env_minimal/bin/activate`
3. Then pip install the requirements via: `pip install -r requirements_minimaltxt`


## Apply NTA to Your Data Set

The script `run_nta.py` can be used to apply NTA to your own data. See the following tables for a description of arguments & example format of input file.

The output file (augmented or reverse translated) will change the column header `aaseq` to `dnaseq` while maintaining the rest of the data.  

#### run_nta.py arguments
| Argument | Type | Description |
|:---------|---------------|-------------|
|input_file| str/bool | name of or path to input data file,  including .csv, i.e. `data.csv`|
|outfile_prefix| str | pefix of output NTA data file, omitting .csv, i.e. `data_nta`. If file is only reverse translated, `_reverse_translated` will be added to prefix. `nta` will be added to prefix if file is augmented|
|aug_size | int | desired length of final augmented data set (number of nucleotide sequences)|
|rev_translate_only| str/bool | if `True`, sequences will only be reverse translated, without augmentation (i.e. for test/val sets). `False` if augmentation desired|

#### Example input file
Ensure that columns named `aaseq` and `target` are present, containing the amino acid sequences (str) and their respective labels (int or float).

| aaseq | target | additional data |
|:---------|---------------|-------------|
|VDVG|1| ...|
|IDGV|	1.4459 | ... |
|LDGV|	1.6901 | ... |
|... | ... | ...|

Example execution using the provided file `Example_NTA_Input_GB1.csv`. Ensure `data/` is unzipped, then run:
```console
python run_nta.py --input_file=data/Example_NTA_Input_GB1.csv --rev_translate_only=False --aug_size=30000 --outfile_prefix=data/gb1_nta

```

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

If you use the the code in this repository for your research, please cite our paper.

```
@article {Minot2022.03.08.483422,
	author = {Minot, Mason and Reddy, Sai T.},
	title = {Nucleotide augmentation for machine learning-guided protein engineering},
	year = {2022},
	doi = {10.1101/2022.03.08.483422},
	URL = {https://www.biorxiv.org/content/early/2022/03/09/2022.03.08.483422},
	journal = {bioRxiv}
}
```
