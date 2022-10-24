#!/bin/bash
cd ..
#run train/eval on amino acids
for m in cnn2 t2
do
    for o in on off
    do
        python main.py --base_model=$m --data_type=gb1 --seq_type=aa --data_set=three_vs_rest --on_off=$o
    done
done

#run train/eval on offline NTA with DNA
for m in cnn2 t2
do
    for t in iterative random codon_shuffle codon_balance 
    do
        for a in none 1 2 3 4 5
        do
            for n in tri_unigram trigram_only unigram
            do
                python main.py --base_model=$m --data_type=gb1 --seq_type=dna --aug_factor=$a  --data_set=three_vs_rest --ngram=$n --seq_file_type=dna --aug_type=$t
            done
        done
    done
done

#run train/eval on online NTA with DNA
for m in cnn2 t2
do
    for t in online online_none online_balance online_shuffle
    do
        for n in tri_unigram trigram_only unigram
        do
            python main.py --base_model=$m --data_type=gb1 --data_set=three_vs_rest --ngram=$n --aug_type=$t
        done
    done
done