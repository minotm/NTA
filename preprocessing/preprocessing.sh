#!/bin/bash
for s in  gb1_train_test_split.py aav_train_test_split.py her2_split_and_imbalance.py 
do
    python $s
done

for a in iterative random codon_balance codon_shuffle online online_balance online_shuffle
do
    for s in gb1_augmentation.py aav_augmentation.py
    python $s --aug_type=$a
done

for a in iterative random codon_balance codon_shuffle
do
    for s in her2_augmentation.py
    python $s --aug_type=$a
done