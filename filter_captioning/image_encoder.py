import torchvision
import torch
import torch.nn as nn

device = 'cuda'
img_encoder_model = torchvision.models.resnet101(pretrained=True)
img_encoder_model = torch.nn.Sequential(*(list(img_encoder_model.children())[:-2]))
# img_encoder_model = img_encoder_model

class Encoder(nn.Module):
  def __init__(self, img_dim, num_proj_layers):
    super().__init__()
    self.resnet_dim = 512
    self.img_dim = img_dim
    self.img_enc = img_encoder_model

    self.num_proj_layers = num_proj_layers

    layers = []
    for i in range(num_proj_layers):
      if i==0:
        layers.append(nn.Sequential(nn.Linear(self.resnet_dim, self.img_dim),
                                    nn.ReLU()
                                    ))
      else:
        layers.append(nn.Sequential(nn.Linear(self.img_dim, self.img_dim),
                                    nn.ReLU()
                                    ))
    self.layers = nn.ModuleList(layers)  
  
  def forward(self, img):
    b = img.shape[0]
    img = self.img_enc(img)
    img = img.view(b,self.resnet_dim, -1).permute(0,2,1)
    img = img.squeeze()
    for i in range(self.num_proj_layers):
      img = self.layers[i](img)
    return img.squeeze()

# img_enc = Encoder(512,1)
# i = {"pixel_values":torch.randn(64,3,224,224)}
# out = img_enc(i)
# print(out.shape)