from huggingface_hub import snapshot_download

repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
local_dir = "/home/model_zoo/LLM/llama2/Llama-2-7b-hf/"  # 本地模型存储的地址
local_dir_use_symlinks = False  # 本地模型使用文件保存，而非blob形式保存
token = "XXX"  # 在hugging face上生成的 access token

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=local_dir_use_symlinks,
    token=token,
    proxies=proxies
)