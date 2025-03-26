from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import SUMMARY_TYPES
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
#1. transform的使用
# 为什么需要tensor数据类型 神经网络基础的数据类型
#绝对路径D:\anaconda\新建文件夹\dataset\train\ants\0013035.jpg
#相对路径dataset/train/ants/0013035.jpg
img_path="dataset/train/ants/0013035.jpg"
img=Image.open(img_path)

writer=SummaryWriter("logs")

tensor_trans=transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img",tensor_img)
writer.close()