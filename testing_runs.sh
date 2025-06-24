## random dataset
python3 main.py --dataset_size=1000 --lsh_size=17 --nb_eLSHes=398 --nb_matches_needed=21 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='dict' --dataset='rand' --nb_queries=200 > rand1000_398.txt
#ls -al heap* >> rand1000_6_90.txt
#rm heap*

python3 main.py --dataset_size=1000 --lsh_size=17 --nb_eLSHes=631 --nb_matches_needed=32 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='dict' --dataset='rand' --nb_queries=200 > rand1000_631.txt
#ls -al heap* >> rand1000_4_90.txt
#rm heap*
# python3 main.py --dataset_size=1000 --lsh_size=18 --nb_eLSHes=1258 --nb_matches_needed=21 --eps_t=85 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='dict' --dataset='rand' --nb_queries=200 > rand1000_4_85.txt
#ls -al heap* >> rand1000_4_85.txt
#rm heap*

python3 main.py --dataset_size=1000 --lsh_size=30 --nb_eLSHes=1000 --nb_matches_needed=16 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='dict' --dataset='synth' --nb_queries=200 > synth1000.txt

python3 main.py --dataset_size=2500 --lsh_size=35 --nb_eLSHes=1150 --nb_matches_needed=16 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='dict' --dataset='synth' --nb_queries=200 > synth2500.txt
#ls -al heap* >> rand10000_6_90.txt
#rm heap*

python3 main.py --dataset_size=5000 --lsh_size=37 --nb_eLSHes=1250 --nb_matches_needed=16 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='dict' --dataset='synth' --nb_queries=200 > synth5000.txt
#ls -al heap* >> rand10000_6_90.txt
#rm heap*

python3 main.py --dataset_size=10000 --lsh_size=22 --nb_eLSHes=631 --nb_matches_needed=30 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='dict' --dataset='rand' --nb_queries=200 > rand10000_6_90.txt
#ls -al heap* >> rand10000_6_90.txt
#rm heap*

python3 main.py --dataset_size=50000 --lsh_size=24 --nb_eLSHes=1800 --nb_matches_needed=20 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='dict' --dataset='rand' --nb_queries=200 > rand50000.txt
python3 main.py --dataset_size=50000 --lsh_size=24 --nb_eLSHes=1500 --nb_matches_needed=21 --eps_t=90 --eps_f=50 --error_rate_percent=10 --map='dict' --dataset='rand' --nb_queries=200 > rand50000_6_90.txt
#ls -al heap* >> rand10000_6_90.txt
#rm heap*

# # Real dataset
# python3 main.py --dataset_size=208 --lsh_size=25 --nb_eLSHes=2000 --nb_matches_needed=30  --error_rate_percent=10 --map='dict' --dataset='nd' --nb_queries=208 > nd_2000_25_30.txt
# ls -al heap* >> nd_2000_25_30.txt
# rm heap*

# python3 main.py --dataset_size=208 --lsh_size=23 --nb_eLSHes=1000 --nb_matches_needed=25  --error_rate_percent=10 --map='dict' --dataset='nd' --nb_queries=208 > nd_1000_23_25.txt
# ls -al heap* >> nd_1000_30_25.txt
# rm heap*
