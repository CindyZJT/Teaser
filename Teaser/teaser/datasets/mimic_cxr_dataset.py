from .base_dataset import BaseDataset
import sys
import random

import os
import json
import torch
import numpy as np
import pickle as pkl
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transq.transforms.utils import MinMaxResize
#from sentence_transformers import SentenceTransformer
from transq.datamodules.tokenizer import Tokenizer

data_dir = "/home/zhaojunting/dataset/mimic_cxr_TranSQ"
threshold = 3
data_name = "mimic"
tokenizer = Tokenizer(data_dir, threshold, data_name)

class MIMIC_Dataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.image_dir = os.path.join(args[0], "images")
        self.image_size = kwargs["image_size"]
        self.max_seq_length = kwargs["max_text_len"]            #80
        self.max_sent_num = kwargs["max_sent_num"]              #25
        self.sent_emb = kwargs["sent_emb"]                      #768
        self.split = kwargs["split"]

        longer = int((1333 / 800) * self.image_size)
        if self.split=="train":
            self.transform = transforms.Compose([
                MinMaxResize(shorter = self.image_size, longer=longer),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))])
        else:
            self.transform = transforms.Compose([
                MinMaxResize(shorter = self.image_size, longer=longer),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))])

        #self.transform = transform
        #self.sent_encoder = SentenceTransformer('all-mpnet-base-v2').encode
        #self.ann = json.loads(open(self.ann_path, 'r').read())

        #self.examples = self.ann[self.split]
        path = os.path.join(kwargs["pre_data_path"], "mimic_{}.pkl".format(self.split))
        # print(path)
        f = open(path, "rb")
        self.logs = pkl.load(f)
        #f = open("./cluster_sentence", "wb")
        #self.centers = pkl.load(f).cpu().numpy()
        #self.cluster_num = 20
        #return self.sentences

    def __getitem__(self, index):
        #example = self.examples[index]
        log = self.logs[index]

        image_id = log['image_id']
        image_path = log['image_path']
        #image
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        
        #topic_seq = log["sent_cls"]
        orig_sent_seq = log["sent_seq"]
        seq_length = log["seq_len"]
        report_ids = log["report_ids"]
        
        common_length = len(log["common_sent_id"])
        specific_length = len(log["specific_sent_id"])
        
        common_sent_id = list(np.zeros(self.max_sent_num, dtype=np.int32))
        common_sent_id[0:common_length] = log["common_sent_id"]
        common_sent_id = np.array(common_sent_id)
        
        specific_sent_id = list(np.zeros(self.max_sent_num, dtype=np.int32))
        specific_sent_id[0:specific_length] = log["specific_sent_id"]
        specific_sent_id = np.array(specific_sent_id)
        
        
        sent_seq = np.zeros((self.max_sent_num, self.sent_emb))                         #句向量序列
        sent_len = orig_sent_seq.shape[0]                                               #句子数
        sent_seq[:sent_len,:] = orig_sent_seq
        


        report_ids_new = np.full((self.max_sent_num, self.max_seq_length),tokenizer.token2idx['<pad>'])

        for idx, r_i in enumerate(report_ids):
            if idx>=self.max_sent_num:
                print("sentence number larger than max_sent_num, with ", len(report_ids))
                continue
            length = min(len(r_i), self.max_seq_length)
            report_ids_new[idx, :length] = np.array(r_i)[:length]                   #按句保存的token_id
        
        seq_length = log["seq_len"]

        #report_masks = np.zeros((self.max_sent_num, self.max_seq_length))
        #report_masks[:, 0] = 1
        #for i in range(seq_length):
        #    text_len = max(len(report_ids[i]), 1)
        #    report_masks[i, : text_len]=1      
            
        #sent_mask = np.zeros((self.max_sent_num))
        #sent_mask[:seq_length+1] = 1 

        sample = {  "image_id": image_id,           #图片id
                    "path": image_path,
                    "image": image,                 #图片（3*N*N）
                    "report_ids": report_ids_new,   #按句拆分的token ids，其中已保证第一位为0
                    #"report_masks": report_masks,   
                    #"sent_label": sent_label,
                    "sent_seq": sent_seq,           #句向量序列，第一项不为全0
                    #"sent_mask": sent_mask,         #句向量mask
                    "seq_len": seq_length,           #句子数
                    # "specific_id_record": [common_sent_id, specific_sent_id],
                    "common_sent_id": common_sent_id,
                    "specific_sent_id": specific_sent_id,
                    "common_length": common_length,
                    "specific_length": specific_length
                 }

        return sample


    def __len__(self):
        return len(self.logs)#//4
        
        # return 640