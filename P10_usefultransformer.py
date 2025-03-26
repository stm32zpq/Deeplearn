from PIL import  Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("dataset/train/ants/0013035.jpg")
print(Image)
trans_totensor = transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("ToTensor",img_tensor)
# 归一化的处理
print(img_tensor[0][0][0])
trans_norm =  transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
# transform_resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_size  =trans_resize(img)
img_size = trans_totensor(img_size)
print(img_size)
#compose -rezise -2说实话就是一个一次执行的过程
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
#RandomCrop
trans_random = transforms.RandomCrop((250,250))
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("img_crop",img_crop,i)
writer.close()