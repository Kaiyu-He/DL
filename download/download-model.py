import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

snapshot_download(repo_id=f"hfl/llama-3-chinese-8b-instruct-v3",
                  ignore_patterns=['*.h5','*.msgpack'],
                  cache_dir="/netcache/hekaiyu/model/llama-3-chinese-8b-instruct-v3", # 缓存目录
                  local_dir=f'/netcache/hekaiyu/model/llama-3-chinese-8b-instruct-v3', # 目标目录
                  local_dir_use_symlinks=False, # 使用软连接
                  resume_download=True,) # 断点续传
