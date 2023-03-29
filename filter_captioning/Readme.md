# Code for modified transformers for image captioning


![](https://i.imgur.com/qSupgh8.png)

A novel idea for image captioning based on having a shared workspace which acts like a filter for filtering the unwanted information from the image. This filter carries only usefull information for image captioning and hence can be fed to the language decoder for generating captions.

All the image embeddings compete against each other to write into the filter and only a certain number of embeddings are allowed to write. This creates a competition among the information content of the embeddings which is governed by the common aim of getting the correct caption.
