import os
import time
import requests
import tarfile
import gzip
from urllib.parse import urlparse
import torchvision  # 导入 torchvision

# 指定数据集存储路径
dataset_root = "./dataset"

# 创建数据集存储目录（如果不存在）
if not os.path.exists(dataset_root):
    os.makedirs(dataset_root)

# 官方 CIFAR-10 数据集下载链接
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
filename = "cifar-10-python.tar.gz"
file_path = os.path.join(dataset_root, filename)

# 定义重试次数和超时时间
max_retries = 5
retry_delay = 5  # 每次重试之间的延迟时间（秒）
timeout = 60  # 超时时间（秒）


def download_file(url, file_path, timeout=60):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()  # 确保请求成功
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # 过滤掉keep-alive的新块
                        f.write(chunk)
            print(f"File downloaded successfully: {file_path}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Download failed.")
                return False


def extract_tar_gz(file_path, extract_to):
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"File extracted successfully: {file_path}")


# 下载并解压数据集
if download_file(url, file_path, timeout):
    extract_tar_gz(file_path, dataset_root)

# 加载训练集
train_set = torchvision.datasets.CIFAR10(root=dataset_root, train=True, download=False)
print("Training set loaded successfully.")

# 加载测试集
test_set = torchvision.datasets.CIFAR10(root=dataset_root, train=False, download=False)
print("Test set loaded successfully.")
