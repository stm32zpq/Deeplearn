import  torchvision
from mpmath.identification import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((375,500)),
    torchvision.transforms.ToTensor()
])
# 准备测试的数据集
# test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.ImageFolder(root="dataset/train", transform=transform)
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
# 测试数据集中第一张图片
img,target=test_data[0]
print(img.shape)
print(target)
writer=SummaryWriter("logs")
step=0
for data in test_loader:
    imgs,targets=data
    print(f"Image tensor shape: {imgs.shape}")
    print(type(imgs))
    print(imgs.shape)
    print(targets)
    writer.add_images("dataloader",imgs,step)
    step = step+1
writer.close()