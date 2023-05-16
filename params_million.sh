#!/bin/bash

for ((i=1; i<=10; i++))
do
    python3 main.py --dataset_size=1000000 --lsh_size=28 --nb_eLSHes=1343 --nb_matches_needed=26 &> million9.txt
done
python3 main.py --dataset_size=1000000 --lsh_size=31 --nb_eLSHes=10738 --nb_matches_needed=25 > million85.txt


python3 main.py --dataset_size=10000 --lsh_size=24 --nb_eLSHes=5012 --nb_matches_needed=33 --eps_t=85 --eps_f=50  > tenthousand85c15.txt
python3 main.py --dataset_size=10000 --lsh_size=21 --nb_eLSHes=631 --nb_matches_needed=22 --eps_t=90 --eps_f=50  > tenthousand9c15.txt

python3 main.py --dataset_size=10000 --lsh_size=23 --nb_eLSHes=3982 --nb_matches_needed=29 --eps_t=85 --eps_f=50  > tenthousand85c13.txt
python3 main.py --dataset_size=10000 --lsh_size=21 --nb_eLSHes=1000 --nb_matches_needed=34 --eps_t=90 --eps_f=50  > tenthousand9c13.txt



