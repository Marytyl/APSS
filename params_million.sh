#!/bin/bash

for ((i=1; i<=10; i++))
do
    python3 main.py --dataset_size=1000000 --lsh_size=28 --nb_eLSHes=1343 --nb_matches_needed=26 &> million9.txt
done
python3 main.py --dataset_size=1000000 --lsh_size=31 --nb_eLSHes=10738 --nb_matches_needed=25 > million85.txt

