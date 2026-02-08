from huggingface_hub import hf_hub_download
import tarfile, os

path = hf_hub_download("ruwwww/batik-processed", filename="batik-256.tar.gz")  # uses HUGGINGFACE_HUB_TOKEN if needed
os.makedirs("data", exist_ok=True)
with tarfile.open(path) as t:
    t.extractall("data")  # result: data/batik-256/images/...