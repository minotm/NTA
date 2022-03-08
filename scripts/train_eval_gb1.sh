#!/bin/bash
cd ..
#run train/eval on amino acids
for m in transformer cnn
do
   python main.py --base_model=$m --data_type=gb1 --seq_type=aa --data_set=three_vs_rest
done

#run train/eval on DNA
for m in transformer cnn
do
    for a in none 1 2 3
    do
        for n in tri_unigram trigram_only unigram
        do
            python main.py --base_model=$m --data_type=gb1 --seq_type=dna --aug_factor=$a  --data_set=three_vs_rest --ngram=$n  --seq_file_type=dna
        done
    done
done
