
# ConTS 
This is our Pytorch implementation for our paper accepted by TOIS 2021:        
>Shijun Li, Wenqiang Lei, Qingyun Wu, Xiangnan He, Peng Jiang, Tat-Seng Chua(2020). Seamlessly Unifying Attributes and Items: Conversational
Recommendation for Cold-Start Users, [Paper in arXiv](https://arxiv.org/abs/2005.12979).

Contributors: Shijun Li.

# Environment Requirement
The code has been tested running under Python 3.7.0. The required packages are as follows:
* torch == 1.0.1
* numpy == 1.15.4
* scipy == 1.1.0
* sklearn == 0.19.2

# Offline Training 
We train an offline Factorization Machine (FM) model following EAR on records of existing users to simultaneously get the embeddings of all attributes, items and existing users. The training process is exactly the same as the process detailed in EAR (https://dl.acm.org/doi/abs/10.1145/3336191.3371769). Specifically, we train the FM by Bayesian Personalized Ranking (BPR), aiming to make it rank user’s preferred items and attributes higher than the others. We do muti-task training, jointly training on the two task of item prediction and attribute prediction. We first train the model for item prediction. After it converges, we continue to optimize the FM for attribute prediction. You can find all the  the code details in https://ear-conv-rec.github.io/manual.html.

# Online CRS
## Datasets
We use three datasets: Yelp, LastFM and Kuaishou. You can download the propocessed Yelp and LastFM datasets at https://drive.google.com/open?id=13WcWe9JthbiSjcGeCWSTB6Ir7O6riiYy. 
Because of confidentiality requirements, we do not offer the raw interaction records in Kuaishou dataset. However, we provide the necessary data for the offline training and online evaluation. All the data files of three datasets are put on Google Drive. You can find the link in '/data/readme.txt'. Sepcifically, these files contains all the positive samples of each user for training, validation and testing,  as well as the relations between all items and attributes. 

## Command
* To run the code, please use the command:
```
python  run_6.py -mt 15 -playby policy -fmCommand 8 -optim SGD -lr 0.01 -decay 0 -TopKTaxo 3 -gamma 0 -strategy maxsim -startFrom 0 -endAt 10000 -eval 1 -initeval 0 -trick 0 -mini 1 -alwaysupdate 1 -upcount 1 -upreg 0.001 -code 0.301 -purpose train -mod ear -upoptim Ada -uplr 0.01
```
Note that since randomless lies in the model (due to the sampling strategy), it's recommended to repeat the experiments 10 times and calculate the average results.

# Reference

Please cite our paper if you use our codes or dataset kindly.
```
@article{li2020seamlessly,
  author    = {Shijun Li and
               Wenqiang Lei and
               Qingyun Wu and
               Xiangnan He and
               Peng Jiang and
               Tat{-}Seng Chua},
  title     = {Seamlessly Unifying Attributes and Items: Conversational Recommendation
               for Cold-Start Users},
  journal   = {The ACM Transactions on Information Systems},
  year      = {2021}
}
```










