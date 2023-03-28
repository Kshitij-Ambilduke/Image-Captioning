from torchvision import transforms
import numpy as np
from PIL import Image
from tokenizers import Tokenizer
import torch 
import random
from torch.utils.data import Dataset, DataLoader


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


tokenizer_path ="/home/ivlabs/Documents/Kshitij/archive/Flickr_tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)
print(tokenizer.get_vocab_size())
tokenizer.enable_padding(pad_id=4)
# image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


class CaptioningDataset(Dataset):
  def __init__(self, split='train'):
    super().__init__()
    self.split=split
    
    data_path = "/home/ivlabs/Documents/Kshitij/archive/captions.txt"
    self.images_path = "/home/ivlabs/Documents/Kshitij/archive/Images/"

    with open(data_path) as f:
      lines = f.readlines()

    lines = lines[1:]
    random.shuffle(lines)
    
    images=[]
    captions=[]

    for some in lines:
      i,c = some.split(',',1)
      images.append(i)
      captions.append(c.rstrip('\n'))
      
    # images = images[1:]
    # captions = captions[1:]
    train_len = 30000

    test_len = (len(captions) - train_len)//2
    
    if self.split=='train':
      images = images[0:train_len]
      captions = captions[0:train_len]

    elif self.split=='test':
      images = images[train_len:train_len+test_len]
      captions = captions[train_len:train_len+test_len]

    elif self.split=='validation':
      images = images[train_len+test_len:train_len+(2*test_len)]
      captions = captions[train_len+test_len:train_len+(2*test_len)]

    self.images = images
    self.captions = captions

  def __len__(self):
    return len(self.images)

  # def __getitem__(self, index):
  #   # print('here')
  #   want_caption = self.captions[index]
  #   want_image = self.images_path + self.images[index]
  #   want_image = Image.open(want_image)
  #   want_image = np.array(want_image.resize((224,224))).reshape(-1,224,224)
  #   if want_image.shape[0]==1:
  #     want_image = np.concatenate((want_image,want_image,want_image),axis=0)
  #   # want_image = want_image.tolist()
  #   # print('here now')
  #   return want_image.tolist(), want_caption 

  def __getitem__(self, index):
    # print('here')
    want_caption = self.captions[index]
    want_image = self.images_path + self.images[index]
    want_img_location = want_image
    want_image = Image.open(want_image)
    # want_image = np.array(want_image.resize((224,224))).reshape(224,224,-1)
    # want_image = np.array(want_image.resize((224,224))).reshape(-1,224,224)
    want_image = preprocess(want_image)
    if want_image.shape[0]==1:
      want_image = np.concatenate((want_image,want_image,want_image),axis=0)
    # want_image = want_image.tolist()
    # print('here now')
    return want_image, want_caption, want_img_location

class MyCollate:
  def __init__(self):
    self.tokenizer = tokenizer

  def __call__(self,batch):
    images=[]
    captions=[]
    image_locations= []
    for i in batch:
      # print(i)
      images.append(i[0])
      captions.append(i[1])
      image_locations.append(i[2])
    
    # print(images)
    # print(captions)
    
    captions = self.tokenizer.encode_batch(captions)

    want_captions = []
    attn = []

    for i in captions:
      # print(i.ids)
      want_captions.append(i.ids)
      attn.append(i.attention_mask)
    # print(want_captions)
    want_captions = torch.Tensor(want_captions).int()
    attn = torch.Tensor(attn)
    # images = torch.Tensor(images)
    images = torch.stack(images)

    return images, want_captions, attn.T, image_locations

## traindataset = CaptioningDataset(split='train')
## train_loader = DataLoader(traindataset, batch_size=32, shuffle=True, collate_fn=MyCollate())
## for i in train_loader:
##     print(i[0].shape)
##     print(i[1].shape)
##     print(i[2].shape)
##     print(i[3])
##     break