from torch.utils.tensorboard import  SummaryWriter
import numpy as np
from PIL import (Image)
writer = SummaryWriter("logs")
img_path= "dataset/train/ants/5650366_e22b7e1065.jpg"
img_PIL=Image.open(img_path)
img_array=np.array(img_PIL)

writer.add_image("test",img_array,28,dataformats='HWC')
#y=x
for i in range(100):
    writer.add_scalar("y=x",i,i)
writer.close()
