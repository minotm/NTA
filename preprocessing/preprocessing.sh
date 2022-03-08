#!/bin/bash
for s in  gb1_train_test_split.py aav_train_test_split.py her2_split_and_imbalance.py augment_gb1.py augment_aav.py augment_her2.py
do
    python $s
done

