import torch
from attention_module import CrossModalAttention, Decoder
import torch.nn as nn 
from einops import repeat
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from image_encoder import Encoder

# Reading and writing refers to reading from the filter and writing to the filter

# Writing: LAYER [N] writes to FILTER
#                                       Q from FILTER
#                                       K and V from LAYER [N]

# Reading: LAYER [N+1] reads from FILTER
#                                       Q from LAYER [N+1] 
#                                       K and V from FILTER
device = 'cuda'

class Encoder_model(nn.Module):
    def __init__(self, num_filters, num_layers, dim, n_heads, pff_dim, dropout, topk, num_proj_layers):
        super().__init__()
        
        self.img_enc = Encoder(dim, num_proj_layers)#.to(device)
        self.topk = topk
        self.num_layers = num_layers
        self.reading_layers = []
        self.writing_layers = []
        self.filter = nn.Parameter(torch.randn(1, num_filters, dim))#.to(device)

        for i in range(num_layers):
            self.reading_layers.append(CrossModalAttention(dim, n_heads, pff_dim, dropout))
            self.writing_layers.append(CrossModalAttention(dim, n_heads, pff_dim, dropout))
        
        self.reading_layers = nn.ModuleList(self.reading_layers)
        self.writing_layers = nn.ModuleList(self.writing_layers)
    
    def forward(self, img):
        
        img = self.img_enc(img)
        b = img.shape[0]
        filters = repeat(self.filter, '() n d -> b n d', b = b)
        # print(filters.shape)

        for i in range(self.num_layers):
            filters = self.writing_layers[i](filters, img, img, None)
            # print(filters.shape)
            img = self.reading_layers[i](img, filters, filters, None, topk=self.topk)
            
        # print(filters.shape)    
        return img, filters

class Captioner(nn.Module):
  def __init__(self, 
               img_dim,         #image encoder
               num_proj_layers, #image encoder
               num_filters,
               num_layers,
               enc_heads,
               enc_pff_dim,
               enc_dropout,
               topk,
               tok_vocab_size,  #output vocab size
               pos_vocab_size,  #max possible length of sentence
               hidden_dim,      
               dec_heads, 
               dec_pff_dim,
               dec_dropout):
    
    super().__init__()
    self.trg_padding_idx = 4
    self.image_encoder = Encoder_model(num_filters, num_layers, img_dim, enc_heads, enc_pff_dim, enc_dropout, topk, num_proj_layers)
    self.language_model = Decoder(tok_vocab_size, pos_vocab_size, hidden_dim, dec_heads, dec_pff_dim, num_layers, dec_dropout)

  def make_trg_mask(self, trg):                                                       # trg = [batch_size, trg_len]                  
    trg_len = trg.shape[1] 
    pad_mask = (trg != self.trg_padding_idx).unsqueeze(1).unsqueeze(2).to(device)   # pad_mask = [batch_size, 1, 1, trg_len]
    sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = device)).bool()   # sub_mask = [trg_len, trg_len]
    trg_mask = pad_mask & sub_mask                                                  # trg_mask = [batch_size, 1, trg_len, trg_len]
    return trg_mask

  def make_src_mask(self, src):                                                       # src = [batch_size, src_len]
        # src_mask = (src != self.src_padding_idx).unsqueeze(1).unsqueeze(2).to(device)   # src_mask = [batch_size, 1, 1, src_len]
        src_mask = torch.ones(src.shape[0],1,1,src.shape[1]).bool().to(device)
        return src_mask

  def forward(self, image, caption, train=True):
    img,filters = self.image_encoder(image)
    if train:
      trg_mask = self.make_trg_mask(caption)
    else:
      # print(caption.shape)
      trg_len = caption.shape[1]
      pad_mask = (caption != self.trg_padding_idx).unsqueeze(1).unsqueeze(2).to(device)   # pad_mask = [batch_size, 1, 1, trg_len]
      sub_mask = torch.ones(trg_len, trg_len).bool().to(device)   # sub_mask = [trg_len, trg_len]
      trg_mask = pad_mask & sub_mask               
    src_mask = self.make_src_mask(img)
    output, attention = self.language_model(caption, trg_mask, img, src_mask)
    return output, attention



# from captioning_dataset import CaptioningDataset, MyCollate
# from torch.utils.data import DataLoader


# traindataset = CaptioningDataset(split='train')
# trainloader = DataLoader(traindataset, batch_size=6, shuffle=True, num_workers=0, collate_fn=MyCollate())
# model = Captioner(img_dim=512,         #image encoder
#                num_proj_layers=1, #image encoder
#                num_filters=4,
#                num_layers=3,
#                enc_heads=2,
#                enc_pff_dim=128,
#                enc_dropout=0.1,
#                topk=3,
#                tok_vocab_size=2706,  #output vocab size
#                pos_vocab_size=5000,  #max possible length of sentence
#                hidden_dim=512,      
#                dec_heads=4, 
#                dec_pff_dim=128,
#                dec_dropout=0.1)
# # enc = Encoder_model(num_filters=4, num_layers=3, dim=512, n_heads=2, pff_dim=512, dropout=0.1, topk=3, num_proj_layers=2)


# for i in trainloader:
#     # img = torch.randn(128,22,64)
#     img = i[0]
#     # print(img.shape)
#     text = i[1]
#     output,_ = model(img,text)
#     # print(img.shape)
#     # print(filt.shape)
#     break