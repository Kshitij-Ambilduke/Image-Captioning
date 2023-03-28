from image_filter_transformer import Captioner
from torch.optim import Adam
import torch.nn as nn
from captioning_dataset import CaptioningDataset, MyCollate
from torch.utils.data import DataLoader, Dataset
import time
import math
import torch


train_dataset = CaptioningDataset()
train_loader = DataLoader(train_dataset, 
                            batch_size=32,
                            collate_fn=MyCollate(),
                            shuffle=False)
criterion = nn.CrossEntropyLoss(ignore_index=4)
model = Captioner(img_dim=512,         #image encoder
               num_proj_layers=1, #image encoder
               num_filters=4,
               num_layers=3,
               enc_heads=2,
               enc_pff_dim=128,
               enc_dropout=0.1,
               topk=3,
               tok_vocab_size=2706,  #output vocab size
               pos_vocab_size=5000,  #max possible length of sentence
               hidden_dim=512,      
               dec_heads=4, 
               dec_pff_dim=128,
               dec_dropout=0.1).to('cuda')
optimizer = Adam(model.parameters(), lr=0.0001)

def train(model, criterion, optimizer, iterator, clip=1, device='cuda'):
  epoch_loss=0
  for data in iterator:
    img = data[0].to(device)
    text = data[1].to(device)
    # img = img.pixel_values.to(device)
    # img = {'pixel_values':img}
    # print(text.shape)
    optimizer.zero_grad()
    model_input_text = text[:,:-1]
    model_output_text = text[:,1:]
    output, _ = model(img, model_input_text)
    output = output.view(-1, output.shape[-1])
    model_output_text = model_output_text.contiguous().view(-1)
    batch_loss = criterion(output, model_output_text.to(device).long())
    batch_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    epoch_loss += batch_loss.item()

  return epoch_loss/len(iterator)

def Epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return (elapsed_mins, elapsed_secs)

import os
MODEL_TYPE = "Filter"
OUTPUT_PATH = f"/home/ivlabs/Documents/Kshitij/thanmay/models/{MODEL_TYPE}"
MODEL_STORE_PATH = os.path.join(OUTPUT_PATH,f"{MODEL_TYPE}_checkpoint_epoch.pth")
EPOCH_SAVE = 4 # Save the model every EPOCH_SAVE epochs
outfile = open(os.path.join(OUTPUT_PATH, f"{MODEL_TYPE}_train_losses.txt"), "w")
outfile.write("Training Loss\tTraining PPL\n")

train_losses = []
valid_losses = []
min_losses = 100
prev_epoch = 1
# min_losses = [float('inf'), float('inf')]
NUM_EPOCHS = 40
start_time = time.time()
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train(model, criterion=criterion, optimizer=optimizer, iterator=train_loader, device='cuda')
    train_losses.append(train_loss)
    if epoch % EPOCH_SAVE == 0:
        torch.save(model.state_dict(), MODEL_STORE_PATH.replace("epoch",str(epoch)))
    elapsed_time = Epoch_time(start_time, time.time())
    print(f"Time taken for epochs {prev_epoch} to {epoch}: {elapsed_time[0]}m {elapsed_time[1]}s")
    start_time = time.time()
    prev_epoch = epoch + 1
    print(f"Training Loss: {train_loss:.4f} ")
    print(f"Training PPL: {math.exp(train_loss):.4f} ")
    outfile.write(f"{train_loss:.4f}\t{math.exp(train_loss):.4f}\n")

outfile.close()