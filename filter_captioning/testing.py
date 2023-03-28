import torch
from image_filter_transformer import Captioner
device = 'cuda'
from torch.utils.data import DataLoader
from captioning_dataset import CaptioningDataset, MyCollate
from tokenizers import Tokenizer

tokenizer_path ="/home/ivlabs/Documents/Kshitij/archive/Flickr_tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)
print(tokenizer.get_vocab_size())
tokenizer.enable_padding(pad_id=4)


test_dataset = CaptioningDataset(split='test')
test_loader = DataLoader(test_dataset, batch_size=46, shuffle=False, collate_fn=MyCollate())

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
               dec_dropout=0.1)
# model.load_state_dict(torch.load('/home/ivlabs/Documents/Kshitij/ResNet_Transformer.pth'))
model = model.to(device)


def testing(model, iterator, tokenizer):
    predictions = []
    locations = []
    captions = []
    model.eval()
    with torch.no_grad():
        for data in enumerate(iterator):
            batch_locations = data[1][-1]
            img = data[1][0]
            text = data[1][1].to(device)
            batch_size = text.shape[0]
            img = img.to(device)
            output, _ = model(img, text, train=True)
            output = torch.softmax(output, dim=-1)
            output = torch.argmax(output, dim=-1)
            predictions.extend(tokenizer.decode_batch(output.tolist()))
            captions.extend(tokenizer.decode_batch(text.tolist()))
            locations.extend(batch_locations)
    
    return predictions, locations, captions


import evaluate

meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')
    
import os
MODEL_TYPE = "Filter"
OUTPUT_PATH = f"/home/ivlabs/Documents/Kshitij/thanmay/models/{MODEL_TYPE}"
MODEL_STORE_PATH = os.path.join(OUTPUT_PATH,f"{MODEL_TYPE}_checkpoint_epoch.pth")
EPOCH_SAVE = 4 # Save the model every EPOCH_SAVE epochs
outfile = open(os.path.join(OUTPUT_PATH, f"{MODEL_TYPE}_scores.txt"), "w")
outfile.write("EPOCH\tBLEU\tMETEOR\tROUGE1\nROUGE2\tROUGE_L\tROUGE_Lsum\n")

NUM_EPOCHS = 40
for epoch in range(EPOCH_SAVE, NUM_EPOCHS + 1, EPOCH_SAVE):
    model.load_state_dict(torch.load(MODEL_STORE_PATH.replace("epoch",str(epoch))))
    predictions, locations, captions = testing(model,test_loader,tokenizer)
    bleu_results = bleu.compute(predictions=predictions, references=captions)
    meteor_results = meteor.compute(predictions=predictions, references=captions)
    rouge_results = rouge.compute(predictions=predictions, references=captions)
    outfile.write(f"{epoch}\t{bleu_results['bleu']}\t{meteor_results['meteor']}\t{rouge_results['rouge1']}\t{rouge_results['rouge2']}\t{rouge_results['rougeL']}\t{rouge_results['rougeLsum']}\n")    
outfile.close()

