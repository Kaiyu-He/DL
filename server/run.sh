conda activate vllm
cd /mnt/usercache/hekaiyu/project
python server/api_vllm.py

conda activate vllm
cd /netcache/hekaiyu/project
python run-vllm.py

cd /netcache/hekaiyu/project
/home/hekaiyu/.conda/envs/sft/bin/python /netcache/hekaiyu/project/prompt/updata_prompt.py