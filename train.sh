cd /netcache/hekaiyu/project
CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch --multi_gpu --num_processes=4 --main_process_port=10425 train/train-hf.py
