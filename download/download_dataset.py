import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
snapshot_download(repo_id="SetFit/mnli",
                  repo_type="dataset",
                  cache_dir='/home/hekaiyu/test/nli-model/dataset/mnli',
                  local_dir='/home/hekaiyu/test/nli-model/dataset/mnli',
                  local_dir_use_symlinks=False,
                  resume_download=True,
                  )
