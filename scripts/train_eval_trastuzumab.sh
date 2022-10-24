#!/bin/bash
cd ..
#run train/eval on amino acids
for m in transformer cnn
do
    python main.py --base_model=$m --data_type=her2 --seq_type=aa
done

#run train/eval on offline NTA with DNA
for m in transformer cnn
do
    for t in iterative random codon_balance codon_shuffle
    do
        for a in none 2 5 10
        do
            for n in tri_unigram trigram_only unigram
            do
                python main.py --base_model=$m --data_type=her2 --seq_type=dna --aug_factor=$a  --ngram=$n --seq_file_type=dna --aug_type=$t
            done
        done
    done
done
