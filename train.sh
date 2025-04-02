conda activate  sft
cd /netcache/hekaiyu/project
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu --num_processes=4 --main_process_port=10908 train/train_hf.py
