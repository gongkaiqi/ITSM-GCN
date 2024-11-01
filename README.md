
## Introduction

This is the PyTorch implementation for our CIKM 2022 paper: 

>CIKM 2022. Kaiqi Gong, Xiao Song, et al. ITSM-GCN: Informative Training Sample Mining for Graph Convolutional Network-based Collaborative Filtering

>Our follow-up research on this paper: KBS 2024. Kaiqi Gong, Xiao Song, et al. HN-GCCF: High-order neighbor-enhanced graph convolutional collaborative filtering

In this work, we propose a training framework ITSM-GCN, which mainly consists of a basic GCN model, our designed ITSM sampling strategy (including a conventional positive sampler, two novel positive samplers, and an improved dynamic negative sampler), and the BPR loss function. In ITSM, the conventional positive sampler randomly selects positive training samples from users’ interaction items. Conversely, our two novel positive samplers augment more potentially informative positive instances for model training according to their respective rules. 


## Instruction

Directly run the file `code/main.py` to get the results on the Gowalla dataset. We also provide our log file about Gowalla in `LightGCN-DNS-1859-si-1884-sc-1895.txt`. Please refer to the hyperparameter settings provided in the original paper for the results of other datasets. 


## Acknowledgement

We refer to the code of [LightGCN](https://github.com/gusye1234/LightGCN-PyTorch). Thanks for their contributions.

 
