from pathlib import Path
import os
import sys

# Device ownership belongs to the launcher; never overwrite its GPU assignment.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

# get current file path
current_file_path = os.path.dirname(os.path.abspath(__file__))
project_root_path = Path(current_file_path).parent

# add source folder to path
sys.path.insert(0, str(project_root_path))

if __name__ == "__main__":
    print(sys.path)
