![Python Version: 3.8.5](https://img.shields.io/badge/Python%20Version-3.8.5-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen)](https://opensource.org/licenses/MIT)

# Nucleotide Augmentation For Machine Learning-Guided Protein Engineering


## Table of contents
1. [Applying NTA to Your Data Set](#apply-nta-to-your-data-set)
2. [Offline Augmentation Script Arguments](#script-arguments)
3. [Input File Format for Offline Augmentation](#input-file-format-for-offline-augmentation)
3. [Online Augmentation Reference](#online-augmentation-reference)
4. [Citation](#citation)


## Applying NTA to Your Data Set

This folder contains several offline **N**ucleo**t**ide **A**ugmentation (**NTA**) methods as well as example code to run online nucleotide augmentation in pytorch as described in Minot & Reddy 2022 [[1](https://www.biorxiv.org/content/10.1101/2022.03.08.483422v1)].


## Offline Augmentation Script Arguments
| Argument | Type | Description |
|:---------|---------------|-------------|
|input_file| str/bool | name of or path to input data file,  including .csv, i.e. `data.csv`|
|outfile_prefix| str | pefix of output NTA data file, omitting .csv, i.e. `data_nta`. If file is only reverse translated, `_reverse_translated` will be added to prefix. `nta` will be added to prefix if file is augmented|
|aug_factor | int | number of times to augment each unique amino acid sequence|
|rev_translate_only| str/bool | if `True`, sequences will only be reverse translated, without augmentation (i.e. for test/val sets). `False` if augmentation desired|
|aug_type | str | augmentation method. options: `random`, `iterative`, `codon_mix`, `reduced_codons`, `online`. Note that `online` will reverse translate aa to dna without augmentation. |

## Input File Format for Offline Augmentation
Ensure that columns named `aaseq` and `target` are present, containing the amino acid sequences (str) and their respective labels (int or float).

The output file (augmented or reverse translated) will change the column header `aaseq` to `dnaseq` while maintaining the rest of the data.  


| aaseq | target | additional data |
|:---------|---------------|-------------|
|VDVG|1| ...|
|IDGV|	1.4459 | ... |
|LDGV|	1.6901 | ... |
|... | ... | ...|

Example execution using the provided file `Example_NTA_Input_GB1.csv`. Ensure `data/` is unzipped, then run:
```console
python run_nta.py --input_file=../data/Example_NTA_Input_GB1.csv --aug_type=random --rev_translate_only=False --aug_factor=10 --outfile_prefix=data/gb1_nta

```

## Online Augmentation Reference
the file `online_nta_example.py` serves as an example of the torch transforms used for online augmentation via codon substitution. The developed online NTA approach assumes input sequences are nucleotides and takes the hyperparameters `subst_frac` and `t_aug`. 

`subst_frac` describes the fraction of the sequence to swap out for synonymous codons.

`prob_of_augmentation` describes a threshold that a random number generated between 0 and 1 must exceed before a sequence is to be modified via online codon substitution. 


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