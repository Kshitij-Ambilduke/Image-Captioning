# Deep learning techniques for automatic Image Captioning

This is the repository that goes with the final year thesis titled "Deep Learning techniques for Automatic Image Captioning" done under the supervision of Dr. Anamika Singh at Visvesvaraya National Institute of Technology, Nagpur.
The repo contains the following implementations of image captioning models on the Flickr8k and Flickr30k datasets.
1. Resnet image encoder and RNN text decoder
2. Resnet image encoder and GRU text decoder
3. Resnet image encoder and LSTM text decoder
4. Resnet image encoder and Transformer text decoder
5. Resnet image encoder with modified transformers 

Among these, the last one i.e. Resnet image encoder with modified transformers is a plausible novelty which results in much faster convergence of the image captioning training process and even the loss convergence is much lower than that of its simple, vanilla transformer counter-part. All these models were examined by calculating the METEOR, BLEU and ROGUE scores for the generated captions.

A part of this work in the form of a literature survey in the field of transformer based image-captioning was presented in the paper titled: "Attending to transformer: A survey on transformer-based image captioning" which was accepted at the 2nd International Conference on the Paradigm shifts in Communication, embedded systems, machine learning and signal processing. 

