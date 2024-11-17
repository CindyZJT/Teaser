import os
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
import lightning.pytorch as pl

# import transq.modules.vision_transformer as vit
import numpy as np
import pickle as pkl
import json
import mmcv
import torchvision.models as models
from einops import rearrange
import joblib
import time
import pandas as pd
from openpyxl import load_workbook


# from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from transq.modules import heads, objectives, mrg_utils
from scipy.optimize import linear_sum_assignment
# import math

# import matplotlib.pyplot as plt


class TransformerSQHparams(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

class TransformerSQ(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.epoch_count = -1
        
        os.makedirs("./_res_dir/ret_logs/", exist_ok=True)
        self.ret_result_file = open("./_res_dir/ret_logs/ret_test.txt", "w")
        

        # self.save_hyperparameters()
        self.transformerSQHparams = TransformerSQHparams(config, tokenizer)
        
        self.tokenizer = tokenizer
        self.sent_len = config["max_sent_num"]
        self.text_size = config["max_text_len"]
        self.image_size = config["max_image_len"]
        self.dataset_name=self.transformerSQHparams.config["datasets"]

        hs = self.transformerSQHparams.config["hidden_size"] 
        
        if self.transformerSQHparams.config["perceiver"]:
            import transq.modules.vision_transformer_perceiver as vit
            
        else:
            import transq.modules.vision_transformer as vit


        self.pos_embeddings = vit.PositionalEncoding(config["hidden_size"], 0.1, self.sent_len)
        self.pos_embeddings.apply(objectives.init_weights)               

        self.pos_embeddings_2 = vit.PositionalEncoding(config["hidden_size"], 0.1, self.text_size)
        self.pos_embeddings_2.apply(objectives.init_weights) 

        self.image_type_embedding = nn.Embedding(2, hs)

        self.vis_dropout = nn.Dropout(0.1)



        '''
        f = open("/home/zhaojunting/sourceCode/updateTranSQ/v8Specific/preprocess/data/sentence_gallery.pkl", "rb")

        gallery = pkl.load(f)
        self.sentence_vectors = gallery.sentence_vectors
        vecs_mean = self.sentence_vectors.mean(axis=1)[:,np.newaxis]
        vec_std  = np.sqrt(self.sentence_vectors.var(axis=1)+1e-6)[:, np.newaxis]
        self.sentence_vectors = (self.sentence_vectors-vecs_mean)/vec_std

        sent_vecs_norm = np.linalg.norm(x = self.sentence_vectors, ord=2, axis = 1, keepdims = True)
        self.sent_vecs_norm = sent_vecs_norm.clip(min=1e-7).reshape(-1,1)
        self.sent_vects = torch.Tensor(self.sentence_vectors/self.sent_vecs_norm)    
        self.sentence_gallery = gallery.sentence_gallery   
        '''       
        
        
        
        # '''
        # 加载common数据
        f1 = open("./preprocess/data/common_sentence_gallery.pkl", "rb")        
        common_gallery = pkl.load(f1)

        self.common_sentence_vectors = common_gallery.sentence_vectors
        common_vecs_mean = self.common_sentence_vectors.mean(axis=1)[:,np.newaxis]
        common_vec_std  = np.sqrt(self.common_sentence_vectors.var(axis=1)+1e-6)[:, np.newaxis]
        self.common_sentence_vectors = (self.common_sentence_vectors-common_vecs_mean)/common_vec_std

        common_sent_vecs_norm = np.linalg.norm(x = self.common_sentence_vectors, ord=2, axis = 1, keepdims = True)
        self.common_sent_vecs_norm = common_sent_vecs_norm.clip(min=1e-7).reshape(-1,1)
        self.sent_vects_common = torch.Tensor(self.common_sentence_vectors/self.common_sent_vecs_norm)    
        self.common_sentence_gallery = common_gallery.sentence_gallery     
        
        
        # 加载specific数据
        f2 = open("./preprocess/data/specific_sentence_gallery.pkl", "rb")
        specific_gallery = pkl.load(f2)
        self.specific_sentence_vectors = specific_gallery.sentence_vectors
        specific_vecs_mean = self.specific_sentence_vectors.mean(axis=1)[:,np.newaxis]
        specific_vec_std  = np.sqrt(self.specific_sentence_vectors.var(axis=1)+1e-6)[:, np.newaxis]
        self.specific_sentence_vectors = (self.specific_sentence_vectors-specific_vecs_mean)/specific_vec_std

        specific_sent_vecs_norm = np.linalg.norm(x = self.specific_sentence_vectors, ord=2, axis = 1, keepdims = True)
        self.specific_sent_vecs_norm = specific_sent_vecs_norm.clip(min=1e-7).reshape(-1,1)
        self.sent_vects_specific = torch.Tensor(self.specific_sentence_vectors/self.specific_sent_vecs_norm)    
        self.specific_sentence_gallery = specific_gallery.sentence_gallery 
        
        
        self.sent_vects = torch.cat([self.sent_vects_common, self.sent_vects_specific], dim=0)
        self.sentence_gallery = self.common_sentence_gallery + self.specific_sentence_gallery  
        # ''''

        if self.transformerSQHparams.config["perceiver"]:
            self.image_query_num = config["image_query_num"]
            self.image_query = nn.Embedding(self.image_query_num, hs)

        self.semantic_query_num = config["semantic_query_num"]
        self.semantic_query_num_common = config["semantic_query_num_common"]
        self.semantic_query_num_specific = config["semantic_query_num_specific"]
        
        self.val_common_To_spefic = 0   # 记录验证过程中，有多少common的query被匹配成 specific
        self.val_specific_To_common = 0  # 记录验证过程中，有多少 specific 的query被匹配成 common
        
        self.test_common = 0   # 记录在测试过程中，common被激活的数量
        self.test_specific = 0   # 记录在测试过程中，common被激活的数量
        
        
        self.semantic_query = nn.Embedding(self.semantic_query_num, hs)
        self.classification_query = nn.Embedding(self.semantic_query_num, hs)
        
        self.train_select_pos_count = np.zeros(self.semantic_query_num)
        self.train_select_neg_count = np.zeros(self.semantic_query_num)        
        self.test_select_pos_count = np.zeros(self.semantic_query_num)        
        self.test_select_neg_count = np.zeros(self.semantic_query_num)
        self.test_select_count = np.zeros(self.semantic_query_num)
        self.path_graph = np.zeros((self.semantic_query_num+2, self.semantic_query_num+2))


        self.test_log = dict()

        self.topic_proj = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, hs),
            )  
        self.topic_clas = nn.Sequential(
                nn.Linear(hs, hs),
                nn.LayerNorm(hs),
                nn.GELU(),
                nn.Linear(hs, self.semantic_query_num),
            )
        self.topic_proj.apply(objectives.init_weights)
        self.topic_clas.apply(objectives.init_weights)

        self.topic_clas2 = nn.Sequential(
                nn.Linear(hs, hs),
                nn.LayerNorm(hs),
                nn.GELU(),
                nn.Linear(hs, 1)
            )
        self.topic_clas2.apply(objectives.init_weights)        
        
        
        self.features_records = dict()   # 用来保存features, 用于T-SNE

        
        if self.transformerSQHparams.config["load_path"] == "":
            print("pretrained: True")
            print(self.transformerSQHparams.config["vit"])
            self.transformer = getattr(vit, self.transformerSQHparams.config["vit"])(
                pretrained=True, config=self.transformerSQHparams.config
            )
        else:
            print("pretrained: False")
            self.transformer = getattr(vit, self.transformerSQHparams.config["vit"])(
                pretrained=False, config=self.transformerSQHparams.config
            )
        


        # ===================== Downstream ===================== #
        if (self.transformerSQHparams.config["load_path"] != "" and not self.transformerSQHparams.config["test_only"]):
            print("test_only",self.transformerSQHparams.config["test_only"])
            print("Load pretrained model from {}".format(self.transformerSQHparams.config["load_path"]))
            ckpt = torch.load(self.transformerSQHparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            

        mrg_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.transformerSQHparams.config["load_path"] != "" and self.transformerSQHparams.config["test_only"]:
            print("Load Pretrained Model From", self.transformerSQHparams.config["load_path"])
            ckpt = torch.load(self.transformerSQHparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
    
    
    
    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        if "mimic" in self.current_tasks:
            ret_t, ret_result = objectives.compute_mimic(self, batch)
            ret.update(ret_t) 
  

        if "iuxray" in self.current_tasks:
            ret_t, ret_result = objectives.compute_iuxray(self, batch)
            ret.update(ret_t) 

        return ret, ret_result 
    
    
    
 
    def training_step(self, batch, batch_idx):
        mrg_utils.set_task(self)
        output, ret_result = self(batch)
        total_loss = output['loss'].mean()
        losses_dict = output['losses_dict']
        self.logger.experiment.add_scalar(f"mimic/trainLoss/total_loss", total_loss, self.global_step)
        self.logger.experiment.add_scalar(f"mimic/trainLoss/sim_loss", losses_dict["sim_loss"], self.global_step)
        self.logger.experiment.add_scalar(f"mimic/trainLoss/nce_loss", losses_dict["nce_loss"], self.global_step)
        self.logger.experiment.add_scalar(f"mimic/trainLoss/cls_loss", losses_dict["cls_loss"].mean(), self.global_step)

        return total_loss
    
    
    def on_train_epoch_start(self):
        print("*" * 100)

        self.train_epoch_start_time = time.time()
        print(time.asctime())


    def on_train_epoch_end(self):
        print("*" * 100)

        self.train_epoch_end_time = time.time()
        print(time.asctime())
        print(self.train_epoch_end_time - self.train_epoch_start_time)
        
        mrg_utils.epoch_wrapup(self)
        class_freq = dict()
        class_freq["class_freq"] = self.train_select_pos_count
        class_freq["neg_class_freq"] = self.train_select_neg_count
        mmcv.dump(class_freq, "./_res_dir/pkl_files/class_freq_store/{}_class_freq_dividually.pkl".format(self.transformerSQHparams.config["exp_name"]))
        mmcv.dump(self.path_graph, "./_res_dir/pkl_files/path_graph_store/{}_path_graph.pkl".format(self.transformerSQHparams.config["exp_name"]))
        print("class_freq.pkl updated.")
        print("path_graph.pkl updated.")
        
        os.makedirs("_res_dir/train_select_pos_count", exist_ok=True)
        excel_file = os.path.join("_res_dir/train_select_pos_count", self.transformerSQHparams.config['exp_name'] + '.csv')
        header = np.array(['epoch']+ [str(i) for i in range(self.train_select_pos_count.shape[0])]).reshape(1,-1)

        record = np.insert(self.train_select_pos_count, 0, self.current_epoch).reshape(1,-1)
        if not os.path.isfile(excel_file):
            with open(excel_file, 'a+') as file:
                np.savetxt(file, header, fmt='%s',delimiter=',')
                np.savetxt(file, record, fmt='%d',delimiter=',')
        else:
            with open(excel_file, 'a+') as file:
                np.savetxt(file, record, fmt='%d',delimiter=',')
        
        self.train_select_pos_count = np.zeros(self.semantic_query_num)
        self.train_select_neg_count = np.zeros(self.semantic_query_num)
        self.path_graph = np.zeros((self.semantic_query_num+2, self.semantic_query_num+2))

    def validation_step(self, batch, batch_idx):
        mrg_utils.set_task(self)
        output, ret_result = self(batch)
        self.save_ret_result(ret_result)

    def on_validation_epoch_end(self):
        print("common的query被匹配成 specific : ", self.val_common_To_spefic)
        print("specific的query被匹配成 common : ", self.val_specific_To_common)
        
        self._print_metrics_to_file('val')

        mrg_utils.epoch_wrapup(self)
        self.ret_result_file.close()
        self.epoch_count+=1
        os.makedirs("./_res_dir/ret_logs/{}/".format(self.transformerSQHparams.config["exp_name"]), exist_ok=True)
        self.ret_result_file = open("./_res_dir/ret_logs/{}/ret_validation_{}.txt".format(self.transformerSQHparams.config["exp_name"], self.current_epoch), "w")
        
        #print(self.test_select_pos_count)
        #print(self.test_select_neg_count)
        #print(self.test_select_count)
        
        self.test_select_pos_count = np.zeros(self.semantic_query_num)
        self.test_select_neg_count = np.zeros(self.semantic_query_num)  
        self.test_select_count = np.zeros(self.semantic_query_num)   
        
        self.val_common_To_spefic = 0  
        self.val_specific_To_common = 0  

    def test_step(self, batch, batch_idx):
        mrg_utils.set_task(self)
        
        '''batch_size设成1
        from thop import profile
        model_FLOPs, model_params = profile(model=self, inputs=(batch,))
        print('model_FLOPs : ', model_FLOPs / 10**9, "G")
        print("model_params : ", model_params/10**6, "M")
        '''
        
        output, ret_result = self(batch)
        ids = output["ids"]
        path = output["path"]
        #print(path)
        attn = output["attn"]
        patch = output["patch"]
        # abs_attn = output["abs_attn"]    # 普通测试不需要，最终结果保存即可
        #sent_feats = output["sent_feats"]
        #print(ret_result[0])
        self.ret_result_file.close()
        self.ret_result_file = open("./_res_dir/ret_logs/{}/ret_test_{}.txt".format(self.transformerSQHparams.config["exp_name"], self.epoch_count), "w")
        # self.save_ret_result(ret_result, ids, path, attn, patch, abs_attn)     # 普通测试不需要，最终结果保存即可
        self.save_ret_result(ret_result, ids, path, attn, patch, None)     
        


        return output #ret

    def on_test_epoch_end(self):
        print("common的query被匹配成 specific", self.val_common_To_spefic)
        print("specific的query被匹配成 common", self.val_specific_To_common)
        self._print_metrics_to_file('test')
        mrg_utils.epoch_wrapup(self)
        self.ret_result_file.close()
        mmcv.dump(self.path_graph, "./_res_dir/pkl_files/path_graph_store/{}_test_path_graph.pkl".format(self.transformerSQHparams.config["exp_name"]))
        # joblib.dump(self.test_log, "./_res_dir/ret_logs/{}/test_result_{}_{}_patch.pkl".format(self.transformerSQHparams.config["exp_name"], self.epoch_count, str(int(self.transformerSQHparams.config["test_threshold_specific"]*100))))        
        joblib.dump(self.test_log, "./_res_dir/ret_logs/{}/test_result_{}_{}.pkl".format(self.transformerSQHparams.config["exp_name"], self.epoch_count, str(int(self.transformerSQHparams.config["test_threshold_specific"]*100))))        
        # joblib.dump(self.test_log, "./_res_dir/ret_logs/{}/test_result_{}_{}_valThreshold.pkl".format(self.transformerSQHparams.config["exp_name"], self.epoch_count, str(int(self.transformerSQHparams.config["test_threshold_specific"]*100))))

        # mmcv.dump(self.path_graph, "./path_graph.pkl")
        # joblib.dump(self.test_log, "./ret_logs/gen/re.pkl")
        
        with open("./_res_dir/ret_logs/{}/test_result_{}_{}.json".format(self.transformerSQHparams.config["exp_name"], self.epoch_count, str(int(self.transformerSQHparams.config["test_threshold_specific"]*100))), 'w') as f:
            json.dump(self.test_log, f)
        # with open("./_res_dir/ret_logs/{}/test_result_{}_{}_valThreshold.json".format(self.transformerSQHparams.config["exp_name"], self.epoch_count, str(int(self.transformerSQHparams.config["test_threshold_specific"]*100))), 'w') as f:
            # json.dump(self.test_log, f)
            
        self.val_common_To_spefic = 0  
        self.val_specific_To_common = 0  

    def configure_optimizers(self):
        return mrg_utils.set_schedule(self)


  
    def matcher(self, sent_feats, topic_preds, sent_embeds, sent_num, prob_mask=False):
        bs, num_query, _ = sent_feats.shape                                         # sent_feats = (b, n_q, dim)
        out_cls   = topic_preds.flatten(0,1).sigmoid()                              # (b*n_q)
        out_feats = F.normalize(sent_feats.flatten(0,1), dim=1)                     # (b*n_q, dim)
        
        tgt_feats = torch.cat([sent_embeds[i, :sent_num[i]] for i in range(bs)])    # (m, dim)
        tgt_cls   = torch.zeros(tgt_feats.shape[0], dtype=torch.long)               # m

        cost_pick = -out_cls.unsqueeze(1)[:, tgt_cls]                               # 被选中则cost = -prob
        cost_sim = torch.cdist(out_feats, tgt_feats, p=2)                           # (b*n_q, m)
        
        if prob_mask == True:
            out_cls_mask = (out_cls<=0.3)
            cost_sim[out_cls_mask] = 100
        
        C = cost_sim + 0.5*cost_pick
        C = C.view(bs, num_query, -1).cpu().detach()                                # (b, n_q, m)
        #print(C[0,:, 0].topk(5, largest=False))
        #print(sent_num)
        #print(C.shape)
        indices = [linear_sum_assignment(c[i].transpose(1,0)) for i, c in enumerate(C.split(list(sent_num.cpu().numpy()), -1))] 
        #print(indices)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for j, i in indices]



  
    def matcher_dividually(self, sent_feats, topic_preds, sent_embeds, sent_num, common_sent_id, common_length, specific_sent_id, specific_length, prob_mask=False):
        bs, num_query, _ = sent_feats.shape                                         # sent_feats = (b, n_q, dim)
                
        # out_cls   = topic_preds.flatten(0,1).sigmoid()                              # (b*n_q)
        # out_feats = F.normalize(sent_feats.flatten(0,1), dim=1)                     # (b*n_q, dim)
        # tgt_feats = torch.cat([sent_embeds[i, :sent_num[i]] for i in range(bs)])    # (m, dim)
        # tgt_cls   = torch.zeros(tgt_feats.shape[0], dtype=torch.long)               # m
        # cost_pick = -out_cls.unsqueeze(1)[:, tgt_cls]                               # 被选中则cost = -prob
        # cost_sim = torch.cdist(out_feats, tgt_feats, p=2)                           # (b*n_q, m)
        # C = cost_sim + 0.5*cost_pick
        # C = C.view(bs, num_query, -1).cpu().detach()                                # (b, n_q, m)
        # indices = [linear_sum_assignment(c[i].transpose(1,0)) for i, c in enumerate(C.split(list(sent_num.cpu().numpy()), -1))] 

     

        out_cls_common = topic_preds[:, :self.semantic_query_num_common].flatten(0,1).sigmoid()
        out_feats_common = F.normalize(sent_feats[:, :self.semantic_query_num_common].flatten(0,1), dim=1)                     # (b*n_q, dim)
        tgt_feats_common = torch.cat([sent_embeds[i][j].unsqueeze(0) for i in range(bs) for j in common_sent_id[i][:common_length[i]]])
        tgt_cls_common = torch.zeros(tgt_feats_common.shape[0], dtype=torch.long)        
        cost_pick_common = -out_cls_common.unsqueeze(1)[:, tgt_cls_common]
        cost_sim_common = torch.cdist(out_feats_common, tgt_feats_common, p=2)      
        C_common = cost_sim_common + self.transformerSQHparams.config["matcher_coe"] * cost_pick_common
        C_common = C_common.view(bs, self.semantic_query_num_common, -1).cpu().detach()
        indices_common = [linear_sum_assignment(c[i].transpose(1,0)) for i, c in enumerate(C_common.split(list(common_length.cpu().numpy()), -1))] 

        
        if sum(specific_length) != 0:
            out_cls_specific = topic_preds[:, self.semantic_query_num_common:].flatten(0,1).sigmoid()
            out_feats_specific = F.normalize(sent_feats[:, self.semantic_query_num_common:].flatten(0,1), dim=1)                     # (b*n_q, dim)
            tgt_feats_specific = torch.cat([sent_embeds[i][j].unsqueeze(0) for i in range(bs) for j in specific_sent_id[i][:specific_length[i]]])
            tgt_cls_specific = torch.zeros(tgt_feats_specific.shape[0], dtype=torch.long)        
            cost_pick_specific = -out_cls_specific.unsqueeze(1)[:, tgt_cls_specific]
            cost_sim_specific = torch.cdist(out_feats_specific, tgt_feats_specific, p=2)      
            C_specific = cost_sim_specific + self.transformerSQHparams.config["matcher_coe"] * cost_pick_specific
            C_specific = C_specific.view(bs, self.semantic_query_num_specific, -1).cpu().detach()
            indices_specific = [linear_sum_assignment(c[i].transpose(1,0)) for i, c in enumerate(C_specific.split(list(specific_length.cpu().numpy()), -1))] 

            indices = []
            for i, _ in enumerate(indices_common):
                common = indices_common[i]    # ([0,1,2,3,4], [x,x,x,x,x])
                specific = indices_specific[i]    # ([0,1,2,3,4], [x,x,x,x,x]) 
                indices.append((np.concatenate([common[0], specific[0]+self.semantic_query_num_common]), np.concatenate([common[1], specific[1]+ self.semantic_query_num_common])))
                
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for j, i in indices]
        else: 
            indices = indices_common    
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for j, i in indices]

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        #print(indices)
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


    def generate_ret_result(self, vis_embeds,sent_feats, semantic_feat,text_ids, indices, gt_indices, topic_preds, batch_size, device):
        batch_indices, src_indices = indices     # 预测的index
        gt_batch_indices, gt_src_indices = gt_indices   # gt index
        sim_set = []
        sent_ret = []
        sent_targ = []
        topic_prob = []
        retrieval_vec=torch.zeros(batch_size, self.semantic_query_num, 768)
        retrieval_sent =torch.zeros(batch_size,1 )

        for i in range(batch_size):
            sim_s = []
            retrieval_vector = torch.zeros(self.semantic_query_num, 768)
            idx = src_indices[(batch_indices==i).nonzero().view(-1)]
            gt_idx = gt_src_indices[(gt_batch_indices==i).nonzero().view(-1)]

            sent_feat_t = sent_feats[i, idx]    # 预测的index对应特征

            topic_pred_t = topic_preds[i, idx].sigmoid().cpu().numpy()   # 预测index对应的预测概率
            topic_gt_t = topic_preds[i, gt_idx].sigmoid().cpu().numpy()   # gt index 对应的预测概率

            sent_feat_t = F.normalize(sent_feat_t, dim=-1).cpu()


            sim = sent_feat_t @ self.sent_vects.transpose(1,0)
            # sim = sent_feat_t @ torch.cat([self.sent_vects_common, self.sent_vects_specific],dim=0).transpose(1,0)

            max_idx = torch.argmax(sim, dim=1).to(device)
            #gt_max_idx = torch.argmax(gt_sim, dim=1).to(device)
            
            for k in range(len(max_idx)):
                
                if idx[k] <= self.semantic_query_num_common:
                    self.test_common += 1
                else:
                    self.test_specific += 1
                
                if idx[k] < self.semantic_query_num_common and max_idx[k] >= self.sent_vects_common.shape[0]:
                    self.val_common_To_spefic += 1
                elif idx[k]>= self.semantic_query_num_common and max_idx[k] < self.sent_vects_common.shape[0]:
                    self.val_specific_To_common += 1


            
            
            for j in range(len(max_idx)):
                retrieval_vector[j,:]=torch.tensor(self.sent_vects[max_idx[j]])
                # retrieval_vector[j,:]=torch.tensor(torch.cat([self.sent_vects_common, self.sent_vects_specific], dim=0)[max_idx[j]])
            sent_ret_t=""
            
            for j in range(len(max_idx)):
                sim_s.append(sim[j][max_idx[j]].cpu().item())
                pred_sent_t = self.sentence_gallery[max_idx[j]]

                pred_sent_t = pred_sent_t.strip(".")+" ."
                sent_ret_t = sent_ret_t + pred_sent_t + " "
            
            sent_targ_t = self.tokenizer.decode_batch(text_ids[i, :, :].int().cpu().numpy())
            sent_targ_str = ""
            for j in range(len(gt_idx)):
                cur_sent = sent_targ_t[j]
                if cur_sent!="":
                    cur_sent = cur_sent.strip()
                    sent_targ_str = sent_targ_str+cur_sent+" . "
            sent_ret_t = sent_ret_t.strip()
            
            sent_targ_str = sent_targ_str.strip()

            sent_ret_t = self.tokenizer.clean_report_mimic_cxr(sent_ret_t)
            sent_targ_str = self.tokenizer.clean_report_mimic_cxr(sent_targ_str)
            #print(len(retrieval_vector))
            retrieval_vec[i,:,:] = retrieval_vector
            #semantic_feats.append(cur_semantic_feat)
            sim_set.append(sim_s)
            sent_ret.append(sent_ret_t)
            sent_targ.append(sent_targ_str)
            # topic_prob.append(((topic_pred_t, gt_idx), (topic_gt_t, gt_idx)))
            topic_prob.append(((topic_pred_t, idx), (topic_gt_t, gt_idx)))

        
        
        ret_result = [sent_ret, sent_targ, sim_set, topic_prob, semantic_feat, retrieval_vec, retrieval_sent]
        return ret_result


    def generate_ret_result_dividually(self, vis_embeds,sent_feats, semantic_feat,text_ids, indices, gt_indices, topic_preds, batch_size, device):
        # 视觉特征；  特征映射； 语义特征； 
        batch_indices, src_indices = indices     # 预测的index
        gt_batch_indices, gt_src_indices = gt_indices   # gt index
        sim_set = []
        sent_ret = []
        sent_targ = []
        topic_prob = []
        retrieval_vec=torch.zeros(batch_size, self.semantic_query_num, 768)
        retrieval_sent =torch.zeros(batch_size,1 )

        for i in range(batch_size):
            sim_s = []
            retrieval_vector = torch.zeros(self.semantic_query_num, 768)
            idx = src_indices[(batch_indices==i).nonzero().view(-1)]
            # common_idx = idx[idx<self.semantic_query_num_common]
            common_index = torch.nonzero(idx < self.semantic_query_num_common).reshape(1,-1)
            specific_index = torch.nonzero(idx >= self.semantic_query_num_common).reshape(1,-1)
            gt_idx = gt_src_indices[(gt_batch_indices==i).nonzero().view(-1)]

            sent_feat_t = sent_feats[i, idx]    # 预测的index对应特征

            topic_pred_t = topic_preds[i, idx].sigmoid().cpu().numpy()   # 预测index对应的预测概率
            topic_gt_t = topic_preds[i, gt_idx].sigmoid().cpu().numpy()   # gt index 对应的预测概率

            sent_feat_t = F.normalize(sent_feat_t, dim=-1).cpu()


            sim = sent_feat_t @ self.sent_vects.transpose(1,0)
            # sim = sent_feat_t @ torch.cat([self.sent_vects_common, self.sent_vects_specific],dim=0).transpose(1,0)

            # max_idx = torch.argmax(sim, dim=1).to(device)

            # max_idx 匹配的方式，限定在自己的库里
            max_idx = torch.zeros(len(idx), dtype=torch.int)
            for k in range(len(idx)):
                if idx[k] < self.semantic_query_num_common:
                    max_idx[k] = torch.argmax(sim[k, :self.sent_vects_common.shape[0]], dim=0).to(device)
                else:
                    max_idx[k] = (torch.argmax(sim[k, self.sent_vects_common.shape[0]:], dim=0) + self.sent_vects_common.shape[0]).to(device)
                    
            

            
            for k in range(len(max_idx)):
                if idx[k] < self.semantic_query_num_common and max_idx[k] >= self.sent_vects_common.shape[0]:
                    self.val_common_To_spefic += 1
                elif idx[k] >= self.semantic_query_num_common and max_idx[k] < self.sent_vects_common.shape[0]:
                    self.val_specific_To_common += 1

            for j in range(len(max_idx)):
                retrieval_vector[j,:]=torch.tensor(self.sent_vects[max_idx[j]])

            sent_ret_t=""
            
            for j in range(len(max_idx)):
                sim_s.append(sim[j][max_idx[j]].cpu().item())
                pred_sent_t = self.sentence_gallery[max_idx[j]]

                pred_sent_t = pred_sent_t.strip(".")+" ."
                sent_ret_t = sent_ret_t + pred_sent_t + " "
            
            sent_ret_t = sent_ret_t.strip()

            sent_targ_t = self.tokenizer.decode_batch(text_ids[i, :, :].int().cpu().numpy())
            sent_targ_str = ""
            for j in range(len(gt_idx)):
                cur_sent = sent_targ_t[j]
                if cur_sent!="":
                    cur_sent = cur_sent.strip()
                    sent_targ_str = sent_targ_str+cur_sent+" . "
            
            sent_targ_str = sent_targ_str.strip()

            sent_ret_t = self.tokenizer.clean_report_mimic_cxr(sent_ret_t)
            sent_targ_str = self.tokenizer.clean_report_mimic_cxr(sent_targ_str)
            #print(len(retrieval_vector))
            retrieval_vec[i,:,:] = retrieval_vector
            #semantic_feats.append(cur_semantic_feat)
            sim_set.append(sim_s)
            sent_ret.append(sent_ret_t)
            sent_targ.append(sent_targ_str)
            # topic_prob.append(((topic_pred_t, gt_idx), (topic_gt_t, gt_idx)))
            topic_prob.append(((topic_pred_t, idx), (topic_gt_t, gt_idx)))

        
        
        ret_result = [sent_ret, sent_targ, sim_set, topic_prob, semantic_feat, retrieval_vec, retrieval_sent]
        return ret_result




    def generate_ret_result_common_only(self, vis_embeds,sent_feats, semantic_feat,text_ids, indices, gt_indices, topic_preds, batch_size, device):
        # 视觉特征；  特征映射； 语义特征； 
        batch_indices, src_indices = indices     # 预测的index
        gt_batch_indices, gt_src_indices = gt_indices   # gt index
        sim_set = []
        sent_ret = []
        sent_targ = []
        topic_prob = []
        retrieval_vec=torch.zeros(batch_size, self.semantic_query_num, 768)
        retrieval_sent =torch.zeros(batch_size,1 )

        for i in range(batch_size):
            sim_s = []
            retrieval_vector = torch.zeros(self.semantic_query_num, 768)
            idx = src_indices[(batch_indices==i).nonzero().view(-1)]
            idx = idx[idx<self.semantic_query_num_common]
            
            gt_idx = gt_src_indices[(gt_batch_indices==i).nonzero().view(-1)]
            gt_idx = gt_idx[gt_idx < self.semantic_query_num_common]
            

            sent_feat_t = sent_feats[i, idx]    # 预测的common index对应特征

            topic_pred_t = topic_preds[i, idx].sigmoid().cpu().numpy()   # 预测common index对应的预测概率
            topic_gt_t = topic_preds[i, gt_idx].sigmoid().cpu().numpy()   # gt index(common部分) 对应的预测概率

            sent_feat_t = F.normalize(sent_feat_t, dim=-1).cpu()

            sim = sent_feat_t @ self.sent_vects_common.transpose(1,0)

            max_idx = torch.argmax(sim, dim=1).to(device)

            for j in range(len(max_idx)):
                retrieval_vector[j,:]=torch.tensor(self.sent_vects_common[max_idx[j]])

            sent_ret_t=""
            
            for j in range(len(max_idx)):
                sim_s.append(sim[j][max_idx[j]].cpu().item())
                pred_sent_t = self.common_sentence_gallery[max_idx[j]]

                pred_sent_t = pred_sent_t.strip(".")+" ."
                sent_ret_t = sent_ret_t + pred_sent_t + " "
            
            sent_ret_t = sent_ret_t.strip()

            sent_targ_t = self.tokenizer.decode_batch(text_ids[i, :, :].int().cpu().numpy())
            sent_targ_str = ""
            for j in range(len(gt_idx)):
                cur_sent = sent_targ_t[j]
                if cur_sent!="":
                    cur_sent = cur_sent.strip()
                    sent_targ_str = sent_targ_str+cur_sent+" . "
            
            sent_targ_str = sent_targ_str.strip()

            sent_ret_t = self.tokenizer.clean_report_mimic_cxr(sent_ret_t)
            sent_targ_str = self.tokenizer.clean_report_mimic_cxr(sent_targ_str)
            #print(len(retrieval_vector))
            retrieval_vec[i,:,:] = retrieval_vector
            #semantic_feats.append(cur_semantic_feat)
            sim_set.append(sim_s)
            sent_ret.append(sent_ret_t)
            sent_targ.append(sent_targ_str)
            # topic_prob.append(((topic_pred_t, gt_idx), (topic_gt_t, gt_idx)))
            topic_prob.append(((topic_pred_t, idx), (topic_gt_t, gt_idx)))

        
        
        ret_result = [sent_ret, sent_targ, sim_set, topic_prob, semantic_feat, retrieval_vec, retrieval_sent]
        return ret_result



    def select_indices_v2(self, sent_feats, indices=None, indices_max=None, batch_size=64, device="cuda:0"):
        batch_set = []
        src_set = []

        for i in range(batch_size):
            sent_feats_t = sent_feats[i]                                    #(query_num, dim)
            if indices!=None:
                pre_select_idxs = indices[1][indices[0]==i]
                sent_feats_t = sent_feats_t[pre_select_idxs]
                if indices_max!=None:
                    first_select = (pre_select_idxs==indices_max[i]).nonzero()[0]
                else:
                    first_select = None

            sent_feats_t = F.normalize(sent_feats_t, dim=-1).cpu()   
            ret = sent_feats_t @ self.sent_vects.transpose(1,0)             #(query_num, gallery_len)
            max_value, max_idx = torch.max(ret, dim=1)                      #(query_num)
            max_value = max_value.to(device)
            
            sent_retrival = self.sent_vects[max_idx].to(device)             #(query_num, dim)
            
            info_matrix = 1-(sent_retrival @ sent_retrival.transpose(1,0))  #(query_num, query_num)
            #info_matrix = info_matrix * max_value                           #relate = distance*similarity
            info_matrix = info_matrix 
  
            select_set = []
            if first_select != None:
                select = first_select
            else:    
                select = torch.argmax(max_value)
            select_set.append(select)
            set_dist = info_matrix[select]
            batch_set.append(i)
            src_set.append(pre_select_idxs[select])
            while set_dist.max()>0.2:                                      #valuable enough for introducing
                select = torch.argmax(set_dist)
                select_set.append(select)
                set_dist = torch.min(set_dist, info_matrix[select])
                batch_set.append(i)
                src_set.append(pre_select_idxs[select])
            #new_indices.append((torch.as_tensor(src_set, dtype=torch.int64), torch.as_tensor(batch_set, dtype=torch.int64)))
            
        #indices = [(torch.as_tensor(batch_set[i], dtype=torch.int64), torch.as_tensor(src_set[i], dtype=torch.int64)) for i in range(len(batch_set))]
        
        indices = (torch.LongTensor(batch_set), torch.LongTensor(src_set))    
        
        return indices


    def select_indices_dividually(self, sent_feats, indices=None, indices_max=None, batch_size=64, device="cuda:0"):
        batch_set = []
        src_set = []
        # sent_vects = torch.cat([self.sent_vects_common, self.sent_vects_specific], dim=0)

        for i in range(batch_size):
            sent_feats_t = sent_feats[i]                                    #(query_num, dim)
            if indices!=None:
                pre_select_idxs = indices[1][indices[0]==i]
                sent_feats_t = sent_feats_t[pre_select_idxs]
                if indices_max!=None:
                    first_select = (pre_select_idxs==indices_max[i]).nonzero()[0]
                else:
                    first_select = None

            sent_feats_t = F.normalize(sent_feats_t, dim=-1).cpu()   
            ret = sent_feats_t @ self.sent_vects.transpose(1,0)             #(query_num, gallery_len)
            max_value, max_idx = torch.max(ret, dim=1)                      #(query_num)
            max_value = max_value.to(device)
            
            sent_retrival = self.sent_vects[max_idx].to(device)             #(query_num, dim)
            
            info_matrix = 1-(sent_retrival @ sent_retrival.transpose(1,0))  #(query_num, query_num)
            #info_matrix = info_matrix * max_value                           #relate = distance*similarity
            info_matrix = info_matrix 
  
            select_set = []
            if first_select != None:
                select = first_select
            else:    
                select = torch.argmax(max_value)
            select_set.append(select)
            set_dist = info_matrix[select]
            batch_set.append(i)
            src_set.append(pre_select_idxs[select])
            while set_dist.max()>0.2:                                      #valuable enough for introducing
                select = torch.argmax(set_dist)
                select_set.append(select)
                set_dist = torch.min(set_dist, info_matrix[select])
                batch_set.append(i)
                src_set.append(pre_select_idxs[select])
            #new_indices.append((torch.as_tensor(src_set, dtype=torch.int64), torch.as_tensor(batch_set, dtype=torch.int64)))
            
        #indices = [(torch.as_tensor(batch_set[i], dtype=torch.int64), torch.as_tensor(src_set[i], dtype=torch.int64)) for i in range(len(batch_set))]
        
        indices = (torch.LongTensor(batch_set), torch.LongTensor(src_set))    
        
        return indices

    def select_indices(self, sent_feats, batch_size=64, device="cuda:0"):
        batch_set = []
        src_set = []
        for i in range(batch_size):
            sent_feats_t = sent_feats[i]                                    #(query_num, dim)
            sent_feats_t = F.normalize(sent_feats_t, dim=-1).cpu()   
            ret = sent_feats_t @ self.sent_vects.transpose(1,0)             #(query_num, gallery_len)
            max_value, max_idx = torch.max(ret, dim=1)                      #(query_num)
            max_value, max_idx = max_value.to(device), max_idx.to(device)
            
            sent_retrival = self.sent_vects[max_idx].to(device)             #(query_num, dim)
            
            info_matrix = 1-(sent_retrival @ sent_retrival.transpose(1,0))  #(query_num, query_num)
            info_matrix = info_matrix * max_value                           #relate = distance*similarity
            #if i==0:
            #    print(info_matrix)
            #    print(max_value)
            select_set = []
            select = torch.argmax(max_value)
            select_set.append(select)
            set_dist = info_matrix[select]
            batch_set.append(i)
            src_set.append(select)
            while set_dist.max()>0.3:                                      #valuable enough for introducing
                select = torch.argmax(set_dist)
                select_set.append(select)
                set_dist = torch.min(set_dist, info_matrix[select])
                batch_set.append(i)
                src_set.append(select)

        indices = (torch.LongTensor(batch_set), torch.LongTensor(src_set))    
        return indices

    def expand_indices(self, sent_feats, indices, batch_size):
        batch_indices, src_indices = indices
        sent_feats = F.normalize(sent_feats, dim=-1)
        new_src = []
        new_batch = []
        for i in range(batch_size):
            idx = src_indices[(batch_indices==i).nonzero().view(-1)]
            sent_feats_t = sent_feats[i, idx]
            sim = (sent_feats[i] @ sent_feats_t.transpose(1,0)).max(-1).values
            #print(idx)
            src = (sim>0.9).nonzero().view(-1).cpu()
            #print(src)
            new_src.append(src) 
            new_batch.append(torch.full_like(src, i)) 
        new_batch_indices = torch.cat(new_batch, dim=0)
        new_src_indices = torch.cat(new_src, dim=0)
        
        return (new_batch_indices, new_src_indices)
    
    
    def expand_indices_dividually(self, sent_feats, indices, batch_size, common_length, specific_length):
        batch_indices, src_indices = indices
        sent_feats = F.normalize(sent_feats, dim=-1)
        new_src = []
        new_batch = []
        for i in range(batch_size):
            # print(i)
            idx = src_indices[(batch_indices==i).nonzero().view(-1)]
            sent_feats_t = sent_feats[i, idx]
            sent_feats_t_commmon = sent_feats_t[0:common_length[i]]
            sent_feats_t_specific = sent_feats_t[common_length[i]:]

            if len(sent_feats_t_commmon) > 0:
                sim_common = (sent_feats[i, 0:self.semantic_query_num_common] @ sent_feats_t_commmon.transpose(1,0)).max(-1).values
                src_common = (sim_common>0.9).nonzero().view(-1).cpu()
            else:
                src_common = torch.Tensor()

            if len(sent_feats_t_specific) > 0:
                sim_specific = (sent_feats[i, self.semantic_query_num_common:] @ sent_feats_t_specific.transpose(1,0)).max(-1).values
                src_specific = (sim_specific>0.9).nonzero().view(-1).cpu() + self.semantic_query_num_common
            else:
                src_specific = torch.Tensor()

            # new_src.append(src) 
            new_src.append(torch.cat([src_common, src_specific], dim=0)) 

            # new_batch.append(torch.full_like(src, i)) 
            new_batch.append(torch.full_like(torch.cat([src_common, src_specific], dim=0), i)) 


        new_batch_indices = torch.cat(new_batch, dim=0).int()
        new_src_indices = torch.cat(new_src, dim=0).int()
        
        return (new_batch_indices, new_src_indices)


    def infer(
        self,
        batch,
        phase = "val",
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        #print(batch)
        #if f"image_{image_token_type_idx - 1}" in batch:
        #    imgkey = f"image_{image_token_type_idx - 1}"
        #else:
        #    imgkey = "image"

        #do_mlm = "_mlm" if mask_text else ""
        #text_ids = batch[f"text_ids{do_mlm}"]
        #text_labels = batch[f"text_labels{do_mlm}"]
        #text_masks = batch[f"text_masks"]
        #text_embeds = self.text_embeddings(text_ids)

        #text
        image_ids = batch["image_id"]
        image_path = batch["path"]
        text_ids = batch["report_ids"].long()

        
        #text_masks = batch["report_masks"]
        #img = batch["image"]


        #image embedding
        if image_embeds is None and image_masks is None:
            img = batch["image"]
            (
                image_embeds,
                pos_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.transformerSQHparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        #sentence embedding
        sent_embeds = batch["sent_seq"].float()
        #sent_masks = batch["sent_mask"]
        sent_num = batch["seq_len"]   # 句子数
        common_sent_id = batch["common_sent_id"]
        specific_sent_id = batch["specific_sent_id"]
        common_length = batch["common_length"]
        specific_length = batch["specific_length"]

        #topic
        #topic_label = batch["sent_label"]

        length = torch.max(batch["seq_len"]).cpu().data

        #print(sent_embeds.size())

        #device = sent_embeds.get_device()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size, sent_length, feat_dim = sent_embeds.size()
        _, image_length, _ = image_embeds.size()
  
  
        ## Main      
        # Step 1: Visual Extractor, : ViT
        x = image_embeds+pos_embeds # (64, 145, 768)
        
        x = self.vis_dropout(x)   
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x)   # _attn: (64, 12, 145, 145)
        x = self.transformer.norm(x)    
        vis_embeds = x
        
        
        if self.transformerSQHparams.config["perceiver"]:

            ### image queries
            image_query = self.image_query(torch.arange(self.image_query_num).to(device))
            x = image_query.repeat(batch_size, 1, 1)
            for i, blk in enumerate(self.transformer.blocks_image):
                x, _attn = blk(x, (vis_embeds+pos_embeds), (vis_embeds))    # _attn: (64, 12, 64, 145)
            
            abs_attn = _attn    
            x = self.transformer.norm_imageQuery(x) 
            image_queried_embeds = x

            # Step 2: Sentence Embedding Generation
            semantic_query = self.semantic_query(torch.arange(self.semantic_query_num).to(device))
            x = semantic_query.repeat(batch_size, 1, 1)
            for i, blk in enumerate(self.transformer.blocks_topic):
                # x, _attn = blk(x, (vis_embeds+pos_embeds), (vis_embeds))
                x, _attn = blk(x, (image_queried_embeds),  (image_queried_embeds))       # _attn: (64, 12, 70, 64)

            x = self.transformer.norm_sent(x)

            semantic_feat = x

            sent_feats = self.topic_proj(x).float()

        else:  
            # Step 2: Sentence Embedding Generation
            semantic_query = self.semantic_query(torch.arange(self.semantic_query_num).to(device))   # (50, 768)
            x = semantic_query.repeat(batch_size, 1, 1)   # (64, 50, 768)
            for i, blk in enumerate(self.transformer.blocks_topic):
                x, _attn = blk(x, (vis_embeds+pos_embeds), (vis_embeds))       # _attn: (64, 12, 70, 145) 
            x = self.transformer.norm_sent(x)

            semantic_feat = x

            sent_feats = self.topic_proj(x).float()
        
        
        # Step 3: assignment of sent_feats prediction
        topic_preds = self.topic_clas2(x).squeeze(-1)


        # matcher : 提取的特征和target之间做sim loss
        # match_idxs = self.matcher(sent_feats, topic_preds, sent_embeds, sent_num)
        match_idxs = self.matcher_dividually(sent_feats, topic_preds, sent_embeds, sent_num, common_sent_id, common_length, specific_sent_id, specific_length)
        
        
        #print(match_idxs)
        
        indices = self._get_src_permutation_idx(match_idxs)
        #include the most similar queries
        if phase=="train":
            # TODO 是否需要将expand分开？ 似乎都有道理
            # expand_indices = self.expand_indices(sent_feats, indices, batch_size)
            expand_indices = self.expand_indices_dividually(sent_feats, indices, batch_size, common_length, specific_length)
        else:
            expand_indices = indices


        if phase == "train":
            # count positive and negative frequents
   
            neg_count = np.ones(self.semantic_query_num)*batch_size
            pos_count = np.zeros(self.semantic_query_num)
            
            for i in range(self.semantic_query_num):
                pos_count[i] += torch.sum(indices[1]==i).cpu().numpy()
                
            neg_count=neg_count-pos_count    
            self.train_select_pos_count = self.train_select_pos_count+pos_count
            self.train_select_neg_count = self.train_select_neg_count+neg_count

        
        ret_result=None
        pred_indices = None
        if phase!="train":  

            
            '''
            # 记录features
            feature_path = "/home/zhaojunting/sourceCode/updateTranSQ/v8Specific/visualization_cases/visualFigures/features_record_" + self.transformerSQHparams.config["exp_name"] + ".pkl"


            if os.path.exists(feature_path):
                f = open(feature_path,"rb")
                features_records = pkl.load(f)
                features_records = torch.cat([features_records, x], dim=0)
                print(features_records.shape)
                mmcv.dump(features_records, feature_path)

            else:
                features_records = x
                mmcv.dump(x, feature_path)    
                 
            '''
            
            # pred_indices = (topic_preds.sigmoid()>0.45).nonzero()
            pred_indices = torch.cat([  \
                topic_preds[:, :self.semantic_query_num_common].sigmoid()> self.transformerSQHparams.config["test_threshold_common"],   \
                topic_preds[:, self.semantic_query_num_common:].sigmoid() > self.transformerSQHparams.config["test_threshold_specific"]],  \
                                     dim=1).nonzero()
            
            os.makedirs("./_res_dir/ret_logs/{}/".format(self.transformerSQHparams.config["exp_name"]), exist_ok=True)

            if self.transformerSQHparams.config["test_only"]:
                pred_indices_log_path = "./_res_dir/ret_logs/{}/pred_indices_log_test_{}.pth".format(self.transformerSQHparams.config["exp_name"], self.epoch_count)
            else:
                pred_indices_log_path = "./_res_dir/ret_logs/{}/pred_indices_log_val_{}.pth".format(self.transformerSQHparams.config["exp_name"], self.current_epoch)

            if os.path.exists(pred_indices_log_path):
                pred_indices_log = torch.load(pred_indices_log_path)
                pred_indices_log["common"] = torch.cat([pred_indices_log["common"],topic_preds[:, :self.semantic_query_num_common].sigmoid()])
                pred_indices_log["specific"] = torch.cat([pred_indices_log["specific"],topic_preds[:, self.semantic_query_num_common:].sigmoid()])
            else:
                pred_indices_log = dict()
                pred_indices_log["common"] = topic_preds[:, :self.semantic_query_num_common].sigmoid()
                pred_indices_log["specific"] = topic_preds[:, self.semantic_query_num_common:].sigmoid()
                
            torch.save(pred_indices_log, pred_indices_log_path)
            
            
            top_k = self.transformerSQHparams.config["topk"]        # top_k = 6
            
            top_indices = topic_preds.sigmoid().topk(top_k, dim=1).indices.view(-1)
            top_batches = torch.Tensor(np.arange(batch_size).repeat(top_k)).to(device)
 
            pred_indices = (torch.cat([pred_indices[:,0], top_batches], dim=0), torch.cat([pred_indices[:,1], top_indices], dim=0))
            
            indices_max = topic_preds.argmax(dim=1)
            
            pred_indices = self.select_indices_dividually(sent_feats, pred_indices, indices_max, batch_size, device)


            neg_count = np.ones(self.semantic_query_num)*batch_size
            pos_count = np.zeros(self.semantic_query_num)
            for i in range(self.semantic_query_num):
                pos_count[i] += torch.sum(pred_indices[1]==i).cpu().numpy()
                
            neg_count=neg_count-pos_count
            self.test_select_pos_count = self.test_select_pos_count+pos_count
            self.test_select_neg_count = self.test_select_neg_count+neg_count
            
            pos_count = np.zeros(self.semantic_query_num)
            for i in range(self.semantic_query_num):
                pos_count[i] += torch.sum(indices[1]==i).cpu().numpy()
                
            self.test_select_count = self.test_select_count+pos_count
            
            # v1：初始版本； 
            ret_result = self.generate_ret_result(vis_embeds,sent_feats, semantic_feat, text_ids, pred_indices, indices, topic_preds, batch_size, device)
            
            # v2: 分开common和specific部分； 
            # ret_result = self.generate_ret_result_dividually(vis_embeds, sent_feats, semantic_feat, text_ids, pred_indices, indices, topic_preds, batch_size, device)
            
            # v3: 只生成common部分结果；
            # ret_result = self.generate_ret_result_common_only(vis_embeds, sent_feats, semantic_feat, text_ids, pred_indices, indices, topic_preds, batch_size, device)
            

        ret = {
            "ids": image_ids,
            "path": image_path,
            "indices": indices,
            "pred_indices": pred_indices,
            "expand_indices": expand_indices,
            "sent_feats": sent_feats,                #预测的sentence feature
            "sent_embeds": sent_embeds,              #ground-truth sentence embedding
            #"sent_masks": sent_masks[:,1:],         #mask of sentence sequence

            #"text_ids": text_ids[:,:,1:],           #List, 每句话的text ids
            #"text_masks": text_masks[:,:,1:],       #List, 每句话的mask
            #"text_feats": text_feats,               #List, 每句话的feature
            #"text_logits": text_logits,             #List, 每句话的预测logits
            
            "topic_preds": topic_preds,
            #"topic_label": topic_label,
            "ret_result" : ret_result,
            "attn": _attn,
            # "abs_attn" : abs_attn,                  # 普通测试不需要，最终结果保存即可
            "patch_index": patch_index,
            "sent_num"   : sent_num,                 #batch中每个report的句子数
            "common_length" :common_length,
            "specific_length" :specific_length
        }
        
        return ret



    def update_topic_path(self, topic):
        start = 0
        #print(topic)
        for i in topic:
            self.path_graph[start, i+1]+=1                          #path_graph[0]:start; path_graph[101]:end; path_graph[i+1]: i-th topic
            start = i+1
        self.path_graph[start, self.semantic_query_num+1]+=1
        #print(self.path_graph.nonzero())

    def save_ret_result(self,ret_result, ids=None, path=None, attn=None, patch=None, abs_attn=None):
        ret_result_zip = zip(ret_result[0], ret_result[1], ret_result[2], ret_result[3], ret_result[4],ret_result[5],ret_result[6],ret_result[7])
        for idx, (sent_ret, sent_tar, s, tp,semantic_feat,retrieval_vec,retrieval_sent, bleu) in enumerate(ret_result_zip):
            self.ret_result_file.write("====\n")
            self.ret_result_file.write("pred: {}\n".format(sent_ret))
            self.ret_result_file.write("targ: {}\n".format(sent_tar))
            #self.ret_result_file.write("vis_embeds: {}\n".format(vis_embeds))
            self.ret_result_file.write("pred prob: {}\n".format(tp[0]))
            self.ret_result_file.write("gt prob: {}\n".format(tp[1]))
            self.ret_result_file.write("retrieval sim: {}\n".format(s))
            self.ret_result_file.write("bleu_1:{} bleu_2:{} bleu_3:{} bleu_4:{}\n".format(bleu[0], bleu[1], bleu[2], bleu[3]))
            self.update_topic_path(tp[1][1])
            if ids!=None:
                t_dict = dict()
                t_dict["path"]      = path[0][idx]
                t_dict["pred"]      = sent_ret
                t_dict["targ"]      = sent_tar
                #t_dict["vis_embeds"] = vis_embeds.tolist()
                t_dict["pred_prob"] = tp[0][0].tolist()
                t_dict["pred_topic"] = tp[0][1].tolist()
                t_dict["gt_prob"]   = tp[1][0].tolist()
                t_dict["gt_topic"] = tp[1][1].tolist()
                # t_dict["attn"]      = attn[idx].tolist()   # 普通测试不需要，最终结果保存即可
                # t_dict["abs_attn"] = abs_attn[idx].tolist()   # 普通测试不需要，最终结果保存即可，仅限Abstractor
                # t_dict["patch"]     = patch[0][idx].tolist()   # 普通测试不需要，最终结果保存即可       
                t_dict["ret_sim"]   = s
                t_dict["bleu"]      = [float(bleu[0]), float(bleu[1]), float(bleu[2]), float(bleu[3])] 
                self.test_log[ids[idx]] = t_dict



    def _print_metrics_to_file(self, phase):
        record = {}
        record.update({'crt_time': time.asctime(time.localtime(time.time()))})
        epoch = -1
        if self.transformerSQHparams.config["load_path"] != "":
            epoch = str(self.epoch_count)+"_test"
        else:
            epoch = self.current_epoch
        record.update({'exp_name': self.transformerSQHparams.config["exp_name"]})
        record.update({'epoch':epoch})
        record.update({'seed': self.transformerSQHparams.config["seed"]})
        record.update({"max_steps" : self.transformerSQHparams.config["max_steps"]})
        record.update({'learning_rate': self.transformerSQHparams.config["learning_rate"]})
        record.update({'precision': self.transformerSQHparams.config["precision"]})
        record.update({'coe_loss':self.transformerSQHparams.config["coe_loss"]})
        record.update({'perceiver':self.transformerSQHparams.config["perceiver"]})
        record.update({'image_query_num':self.transformerSQHparams.config["image_query_num"]})
        record.update({'test_threshold_common': self.transformerSQHparams.config["test_threshold_common"]})
        record.update({'test_threshold_specific': self.transformerSQHparams.config["test_threshold_specific"]})
        # record.update({'COE_cost_pick': self.transformerSQHparams.config["COE_cost_pick"]})
        record.update({'matcher_coe': self.transformerSQHparams.config["matcher_coe"]})
        record.update({'topk': self.transformerSQHparams.config["topk"]})
        record.update({'generate_version': 'v1'})
        
        record.update({'phase': phase})
        record.update({'val_mimic_bleu1':self.val_mimic_bleu1.compute().item()})
        record.update({'val_mimic_bleu2':self.val_mimic_bleu2.compute().item()})
        record.update({'val_mimic_bleu3':self.val_mimic_bleu3.compute().item()})
        record.update({'val_mimic_bleu4':self.val_mimic_bleu4.compute().item()})
        record.update({'val_mimic_meteor':self.val_mimic_mrg_score.compute()["METEOR"]})
        record.update({'val_mimic_rougeL':self.val_mimic_mrg_score.compute()["ROUGE_L"]})
        record.update({'val_common_To_spefic': self.val_common_To_spefic})
        record.update({'val_specific_To_common': self.val_specific_To_common})
        
        record.update({'test_common': self.test_common})
        record.update({'test_specific': self.test_specific})
        
        

        os.makedirs(self.transformerSQHparams.config['record_path'], exist_ok=True)
        record_path = os.path.join(self.transformerSQHparams.config['record_path'], self.transformerSQHparams.config['exp_name'] + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        
        record = pd.DataFrame.from_dict(record, orient='index').T
        record_table = pd.concat([record_table, record], ignore_index=True)

        record_table.to_csv(record_path, index=False)
        



if __name__=="main":
    model = TransformerSQ()