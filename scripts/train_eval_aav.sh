#!/bin/bash
cd ..
#run train/eval on amino acids
for m in cnn transformer
do
    python main.py --base_model=$m --data_type=aav --seq_type=aa --data_set=seven_vs_rest
done

#run train/eval on DNA
for m in cnn transformer
do
    for a in none 1 2 3
    do
        for n in tri_unigram trigram_only unigram
        do
            python main.py --base_model=$m --data_type=aav --seq_type=dna --aug_factor=$a  --data_set=seven_vs_rest --ngram=$n --seq_file_type=dna
        done
    done
done
