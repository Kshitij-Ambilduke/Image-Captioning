# Deep learning techniques for automatic Image Captioning

This is the repository that goes with the final year thesis titled "Deep Learning techniques for Automatic Image Captioning" done under the supervision of Dr. Anamika Singh at Visvesvaraya National Institute of Technology, Nagpur.
The repo contains the following implementations of image captioning models on the Flickr8k and Flickr30k datasets.
1. [Resnet image encoder and RNN text decoder](https://github.com/Kshitij-Ambilduke/Image-Captioning/blob/main/%5BFYP%5D_Resnet_RNN.ipynb)
2. [Resnet image encoder and GRU text decoder](https://github.com/Kshitij-Ambilduke/Image-Captioning/blob/main/%5BFYP%5D_Resnet_GRU.ipynb)
3. [Resnet image encoder and LSTM text decoder](https://github.com/Kshitij-Ambilduke/Image-Captioning/blob/main/%5BFYP%5D_Resnet_LSTM.ipynb)
4. [Resnet image encoder and Transformer text decoder](https://github.com/Kshitij-Ambilduke/Image-Captioning/blob/main/%5BFYP%5D_Resnet_Transformers.ipynb)
5. [Resnet image encoder with modified transformers ](https://github.com/Kshitij-Ambilduke/Image-Captioning/tree/main/filter_captioning)

Among these, the last one i.e. Resnet image encoder with modified transformers is a plausible novelty which results in much faster convergence of the image captioning training process and even the loss convergence is much lower than that of its simple, vanilla transformer counter-part. All these models were examined by calculating the METEOR, BLEU and ROGUE scores for the generated captions.

A part of this work in the form of a literature survey in the field of transformer based image-captioning was presented in the paper titled: "Attending to transformer: A survey on transformer-based image captioning" which was accepted at the 2nd International Conference on the Paradigm shifts in Communication, embedded systems, machine learning and signal processing.


## Plots


| Scores                               | Loss Curves                          |
| ------------------------------------ | ------------------------------------ |
| ![](https://i.imgur.com/qktfuON.png) | ![](https://i.imgur.com/zGZNAsq.png) |
| ![](https://i.imgur.com/yoYtFds.png) | ![](https://i.imgur.com/N8f48Ra.png) |
| ![](https://i.imgur.com/otftfjx.png) | ![](https://i.imgur.com/hQm0Mci.png) |
| ![](https://i.imgur.com/QjU37kJ.png) | ![](https://i.imgur.com/YxUmPae.png) |

## Sample Outputs



| ![](https://i.imgur.com/FfzKGDi.png) | Prediction: snowboarder in all red coat slides riding down the slope hill . . . . . . . . . . . . . . . . . .  <br /> <br /> Reference: A snowboarder wearing a red jacket is boarding down the snowy hill .|
| ------------------------------------ |--------|

<!-- | ![](https://i.imgur.com/FfzKGDi.png) | Prediction: snowboarder in all red coat slides riding down the slope hill .  <br /> <br /> Reference: A snowboarder wearing a red jacket is boarding down the snowy hill .|
| ------------------------------------ |--------| -->


## Contributors

* Kshitij Ambilduke
* Thanmay Jayakumar


