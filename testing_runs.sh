## random dataset
#python3 main.py --dataset_size=1000 --lsh_size=17 --nb_eLSHes=398 --nb_matches_needed=21 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='omap' --dataset='rand' --nb_queries=200 > rand1000_398.txt
#ls -al heap* >> rand1000_6_90.txt
#rm heap*

#python3 main.py --dataset_size=1000 --lsh_size=17 --nb_eLSHes=631 --nb_matches_needed=32 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='omap' --dataset='rand' --nb_queries=200 > rand1000_631.txt
#ls -al heap* >> rand1000_4_90.txt
#rm heap*
# python3 main.py --dataset_size=1000 --lsh_size=18 --nb_eLSHes=1258 --nb_matches_needed=21 --eps_t=85 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='omap' --dataset='rand' --nb_queries=200 > rand1000_4_85.txt
#ls -al heap* >> rand1000_4_85.txt
#rm heap*

python3 main.py --dataset_size=1000 --lsh_size=17 --nb_eLSHes=500 --nb_matches_needed=12 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='omap' --dataset='synth' --nb_queries=200 > synth1000.txt
python3 main.py --dataset_size=2500 --lsh_size=20 --nb_eLSHes=600 --nb_matches_needed=10 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='omap' --dataset='synth' --nb_queries=200 > synth2500.txt
python3 main.py --dataset_size=5000 --lsh_size=21 --nb_eLSHes=700 --nb_matches_needed=13 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='omap' --dataset='synth' --nb_queries=200 > synth5000.txt
python3 main.py --dataset_size=10000 --lsh_size=23 --nb_eLSHes=800 --nb_matches_needed=10 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='omap' --dataset='synth' --nb_queries=200 > synth10000.txt
python3 main.py --dataset_size=25000 --lsh_size=25 --nb_eLSHes=800 --nb_matches_needed=10 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='omap' --dataset='synth' --nb_queries=200  > synth25000.txt

python3 main.py --dataset_size=1000 --lsh_size=17 --nb_eLSHes=500 --nb_matches_needed=12 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='omap' --dataset='rand' --nb_queries=200 > rand1000.txt
python3 main.py --dataset_size=2500 --lsh_size=20 --nb_eLSHes=600 --nb_matches_needed=10 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='omap' --dataset='rand' --nb_queries=200 > rand2500.txt
python3 main.py --dataset_size=5000 --lsh_size=21 --nb_eLSHes=700 --nb_matches_needed=13 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='omap' --dataset='rand' --nb_queries=200 > rand5000.txt
python3 main.py --dataset_size=10000 --lsh_size=23 --nb_eLSHes=800 --nb_matches_needed=10 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='omap' --dataset='rand' --nb_queries=200 > rand10000.txt
python3 main.py --dataset_size=25000 --lsh_size=25 --nb_eLSHes=800 --nb_matches_needed=10 --eps_t=90 --eps_f=50 --error_dist_file='amey_error_dist.json' --map='omap' --dataset='rand' --nb_queries=200  > rand25000.txt
