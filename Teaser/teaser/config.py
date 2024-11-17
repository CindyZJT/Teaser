# 运行命令
# CUDA_VISIBLE_DEVICES=2 python run.py with task_train_mimic mimic_20w_5020_full_matcher_coe0
# CUDA_VISIBLE_DEVICES=2 python run.py with task_train_mimic mimic_20w_5020_full_matcher_coe100




from sacred import Experiment



ex = Experiment("TranSQ")

def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "mimic": 0,
    }
    
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "vilt"
    datasets = []
    loss_names = _loss_names({"itm": 1, "mlm": 1})

    seed = 0
    batch_size = 64  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    #train_transform_keys = ["pixelbert"]
    train_transform_keys = ["pixelbert_randaug"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    vis_feature_size=384
    max_image_len = 300
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 80
    max_sent_num = 25
    sent_emb = 768
    tokenizer = "bert-base-uncased"

    
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "vit_base_patch32_384"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1

    resnet = "resnet50"

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-2
    #weight_decay = 0.01
    weight_decay = 1e-5
    decay_power = 1
    max_epoch = 500
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 1e-7
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0

    # below params varies with the environment
    data_root = ""
    ann_path = ""

    log_dir = "_res_dir/result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    pre_data_path = ""
    
    # load_path ="./model/TranSQ-mimic.ckpt"
    load_path =""
    record_path = "_res_dir/record_path"
    

    num_workers = 0
    precision = 16
    backbone="ViT"
    

    
    
    # 调参
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0}
    
    # test_threshold = 0.45   # first step in testing 
    test_threshold_common = 0.45
    test_threshold_specific = 0.45

    perceiver = False
    image_query_num = 0
    
    semantic_query_num = 50
    semantic_query_num_common = 30
    semantic_query_num_specific = 20
  
  
    topk = 6
    test_only = False
    test_epoch = [-1]


@ex.named_config
def task_train_mimic():
    exp_name = "train_mimic_vit"
    datasets = ["mimic"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"mimic": 1})
    batch_size = 64
    seed = 10086
    max_image_len = 300
    max_epoch = 100
    max_steps = None
    warmup_steps = 0.02
    semantic_query_num = 50
    semantic_query_num_common = 30
    semantic_query_num_specific = 20
    get_recall_metric = False
    draw_false_text = 15
    learning_rate = 1e-4
    data_root = "/home/zhaojunting/dataset/mimic_cxr_TranSQ"

    
    pre_data_path = "/home/zhaojunting/sourceCode/updateTranSQ/v8Specific/preprocess/data"
    num_gpus = 1
    num_nodes = 1
    per_gpu_batchsize = 64
    val_check_interval = 1.0
    backbone="ViT"
    fast_dev_run = False
    
    
    # 调参
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0}
    
    # test_threshold = 0.45   # first step in testing 
    test_threshold_common = 0.45
    test_threshold_specific = 0.45

    perceiver = False
    image_query_num = 0
    
    semantic_query_num = 50
    semantic_query_num_common = 30
    semantic_query_num_specific = 20
  
    topk = 6
    test_only = False
    test_epoch = [-1]


@ex.named_config
def task_train_mimic_cnn():
    exp_name = "train_mimic_cnn"
    datasets = ["mimic"]
    train_transform_keys = ["pixelbert_randaug"]
    #loss_names = _loss_names({"mimic": 1})
    loss_names = _loss_names({"mimic": 1})
    batch_size = 64
    max_image_len = 300
    vis_feature_size=12
    max_epoch = 100
    max_steps = None
    warmup_steps = 0.02
    patch_size = 12
    get_recall_metric = False
    draw_false_text = 15
    learning_rate = 1e-4
    data_root = "/home/zhaojunting/dataset/mimic_cxr_TranSQ"
    pre_data_path = "/home/zhaojunting/sourceCode/updateTranSQ/v8Specific/preprocess/data"
    num_gpus = 1
    per_gpu_batchsize = 64
    val_check_interval = 1.0
    backbone="CNN"


#  ↓ ---------------------↓-------------------↓-----------------↓---------------↓---------------------------------
#  ↓ ---------------------↓-------------------↓-----------------↓---------------↓---------------------------------
#  ↓ ---------------------↓-------------------↓-----------------↓---------------↓---------------------------------
#  ↓ ---------------------↓-------------------↓-----------------↓---------------↓---------------------------------
   


    
# final result
@ex.named_config
def mimic_20w_5020():
    seed = 10086
    exp_name = "mimic_20w_5020"
    max_steps = 200000
    
    # --------------------  调参  -----------------------
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0}

    perceiver = False
    image_query_num = 0
    
    semantic_query_num = 70
    semantic_query_num_common = 50
    semantic_query_num_specific = 20
    
    test_threshold_common = [0.45]
    test_threshold_specific = [0.45]

    test_only = True
    test_epoch = [38]
 
    
# final result
@ex.named_config
def mimic_20w_5020_info():
    seed = 10086
    exp_name = "mimic_20w_5020_info"
    max_steps = 200000
    
    # --------------------  调参  -----------------------
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = False
    image_query_num = 0
    
    semantic_query_num = 70
    semantic_query_num_common = 50
    semantic_query_num_specific = 20
    
    test_threshold_common = [0.45]
    test_threshold_specific = [0.45]  

    test_only = True
    test_epoch = [16]


 
# final版本
@ex.named_config
def mimic_20w_5020_full():
    seed = 10086
    exp_name = "mimic_20w_5020_full"
    max_steps = 200000
    # batch_size = 1 
    # per_gpu_batchsize = 1

    
    # --------------------  调参  -----------------------
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 64
    
    semantic_query_num = 70
    semantic_query_num_common = 50
    semantic_query_num_specific = 20
    
    test_threshold_common = [0.55]   # 0.45， 0.5， 0.4，0.55
    test_threshold_specific = [0.4, 0.5, 0.6, 0.45, 0.55]    
    
    test_only = True
    test_epoch = [41]
    


# final版本_修改匈牙利匹配系数
@ex.named_config
def mimic_20w_5020_full_matcher_coe0():
    seed = 10086
    exp_name = "mimic_20w_5020_full_matcher_coe0"
    max_steps = 200000
    
    # --------------------  调参  -----------------------
    matcher_coe = 0   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 64
    
    semantic_query_num = 70
    semantic_query_num_common = 50
    semantic_query_num_specific = 20
    
    test_threshold_common = [0.45]   # 0.45， 0.5， 0.4，0.55
    test_threshold_specific = [0.45]    
    
    test_only = True
    test_epoch = list(range(0,47))
    
# final版本_修改匈牙利匹配系数 0.25
@ex.named_config
def mimic_20w_5020_full_matcher_coe025():
    seed = 10086
    exp_name = "mimic_20w_5020_full_matcher_coe025"
    max_steps = 200000
    # batch_size = 1 
    # per_gpu_batchsize = 1

    
    # --------------------  调参  -----------------------
    matcher_coe = 0.25   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 64
    
    semantic_query_num = 70
    semantic_query_num_common = 50
    semantic_query_num_specific = 20
    
    test_threshold_common = [0.45]   
    test_threshold_specific = [0.45]    
    
    test_only = True
    test_epoch = list(range(10,40))


# final版本_修改匈牙利匹配系数 0.75
@ex.named_config
def mimic_20w_5020_full_matcher_coe075():
    seed = 10086
    exp_name = "mimic_20w_5020_full_matcher_coe075"
    max_steps = 200000
    
    # --------------------  调参  -----------------------
    matcher_coe = 0.75   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 64
    
    semantic_query_num = 70
    semantic_query_num_common = 50
    semantic_query_num_specific = 20
    
    test_threshold_common = [0.45]   
    test_threshold_specific = [0.45]    
    
    test_only = True
    test_epoch = [32,34,36,38,40]


# final版本_修改匈牙利匹配系数 1.0
@ex.named_config
def mimic_20w_5020_full_matcher_coe100():
    seed = 10086
    exp_name = "mimic_20w_5020_full_matcher_coe100"
    max_steps = 200000
    
    # --------------------  调参  -----------------------
    matcher_coe = 1.0  # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 64
    
    semantic_query_num = 70
    semantic_query_num_common = 50
    semantic_query_num_specific = 20
    
    test_threshold_common = [0.45]   
    test_threshold_specific = [0.45]    
    
    test_only = True
    test_epoch = list(range(16,20))


# final版本, 换了种子
@ex.named_config
def mimic_20w_5020_full_final_1215():
    seed = 1215
    exp_name = "mimic_20w_5020_full_final_1215"
    max_steps = 200000
    
    # --------------------  调参  -----------------------
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 64
    
    semantic_query_num = 70
    semantic_query_num_common = 50
    semantic_query_num_specific = 20
    
    test_threshold_common = [0.45]
    test_threshold_specific = [0.45]
    
    # test_only = True
    # test_epoch = list(range(30,40))
    load_path = "/home/zhaojunting/sourceCode/updateTranSQ/v8Specific/_res_dir/checkpoints/mimic_20w_5020_full_final_1215/epoch=37_seed1215.ckpt"
    

# final版本, 换了种子
@ex.named_config
def mimic_20w_5020_full_final_20201015():
    seed = 20201015
    exp_name = "mimic_20w_5020_full_final_20201015"
    max_steps = 200000
    
    # --------------------  调参  -----------------------
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 64
    
    semantic_query_num = 70
    semantic_query_num_common = 50
    semantic_query_num_specific = 20
    
    # test_threshold_common = [0.45]
    # test_threshold_specific = [0.45]
    
    # test_only = True
    # test_epoch = list(range(30,40))
    # load_path = "/home/zhaojunting/sourceCode/updateTranSQ/v8Specific/_res_dir/checkpoints/mimic_20w_5020_full_final_1215/epoch=37_seed1215.ckpt"
    

# final版本, 换了种子
@ex.named_config
def mimic_20w_5020_full_final_0127():
    seed = 127
    exp_name = "mimic_20w_5020_full_final_0127"
    max_steps = 200000
    
    # --------------------  调参  -----------------------
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 64
    
    semantic_query_num = 70
    semantic_query_num_common = 50
    semantic_query_num_specific = 20
    
    test_threshold_common = [0.45]   # 0.45， 0.5， 0.4，0.55
    test_threshold_specific = [0.45]
    
    test_only = True
    test_epoch = list(range(43,46))

    
# --------------------------------------------------
# perciever 数量讨论
# limit_val_batches = 0.1,    # 使用多少验证集做验证
@ex.named_config
def mimic_20w_5020_full_per0():
    seed = 10086
    exp_name = "mimic_20w_5020_full_per0"
    max_steps = 200000
    
    batch_size = 4
    per_gpu_batchsize = 4
    

    # --------------------  调参  -----------------------
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 0
    
    semantic_query_num = 70
    semantic_query_num_common = 50
    semantic_query_num_specific = 20
    
    # test_threshold_common = [0.45]
    # test_threshold_specific = [0.375]  

    # test_only = True
    # test_epoch = list(range(0,47))
    

# perciever 数量讨论
# limit_val_batches = 0.1,    # 使用多少验证集做验证
@ex.named_config
def mimic_20w_5020_full_per32():
    seed = 10086
    exp_name = "mimic_20w_5020_full_per32"
    max_steps = 200000
    
    # --------------------  调参  -----------------------
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 32
    
    semantic_query_num = 70
    semantic_query_num_common = 50
    semantic_query_num_specific = 20
    
    test_threshold_common = [0.45]
    test_threshold_specific = [0.375]  

    test_only = True
    test_epoch = list(range(0,47))


# perciever 数量讨论
# limit_val_batches = 0.1,    # 使用多少验证集做验证
@ex.named_config
def mimic_20w_5020_full_per128():
    seed = 10086
    exp_name = "mimic_20w_5020_full_per128"
    max_steps = 200000
    batch_size = 32
    per_gpu_batchsize = 32
    
    # --------------------  调参  -----------------------
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 128
    
    semantic_query_num = 70
    semantic_query_num_common = 50
    semantic_query_num_specific = 20
    
    test_threshold_common = [0.45]
    test_threshold_specific = [0.45]  

    test_only = True
    test_epoch = list(range(0,47))


# --------------------------------------------------
# common/specific 数量讨论
# final版本
@ex.named_config
def mimic_20w_5030_full():
    seed = 10086
    exp_name = "mimic_20w_5030_full"
    max_steps = 200000
    
    # --------------------  调参  -----------------------
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 64
    
    semantic_query_num = 80
    semantic_query_num_common = 50
    semantic_query_num_specific = 30
    
    test_threshold_common = [0.45]
    test_threshold_specific = [0.45]  

    test_only = True
    test_epoch = [41]
    
    
    
    
# --------------------------------------------------
# common/specific 数量讨论
# final版本
@ex.named_config
def mimic_20w_5010_full():
    seed = 10086
    exp_name = "mimic_20w_5010_full"
    max_steps = 200000
    
    # --------------------  调参  -----------------------
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 64
    
    semantic_query_num = 60
    semantic_query_num_common = 50
    semantic_query_num_specific = 10
    
    test_threshold_common = [0.45]
    test_threshold_specific = [0.45]  

    test_only = True
    test_epoch = [41]
    
    
    
# --------------------------------------------------
# common/rare 数量讨论
# final版本
@ex.named_config
def mimic_20w_2520_full():
    seed = 10086
    exp_name = "mimic_20w_2520_full"
    max_steps = 200000
    
    # --------------------  调参  -----------------------
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 64
    
    semantic_query_num = 45
    semantic_query_num_common = 25
    semantic_query_num_specific = 20
    
    test_threshold_common = [0.45]
    test_threshold_specific = [0.45]  

    test_only = True
    test_epoch = [10,11]
    

# --------------------------------------------------
# common/rare 数量讨论
# final版本
@ex.named_config
def mimic_20w_7520_full():
    seed = 10086
    exp_name = "mimic_20w_7520_full"
    max_steps = 200000
    
    # --------------------  调参  -----------0------------
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 64
    
    semantic_query_num = 95
    semantic_query_num_common = 75
    semantic_query_num_specific = 20
    
    test_threshold_common = [0.45]
    test_threshold_specific = [0.45]  

    test_only = True
    test_epoch = [7]


 
# final版本
@ex.named_config
def tmp():
    seed = 10086
    exp_name = "tmp"
    max_steps = 200000
    max_epoch = 1

    
    batch_size = 64
    per_gpu_batchsize = 64  # you should define this manually with per_gpu_batch_size=#

    
    # --------------------  调参  -----------------------
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 64
    
    semantic_query_num = 95
    semantic_query_num_common = 75
    semantic_query_num_specific = 20
    
    # test_threshold_common = [0.4, 0.45, 0.5]
    # test_threshold_specific = [0.4, 0.45, 0.5]  

    test_only = False
    # test_epoch = [41]
    
    



# final版本
@ex.named_config
def trainingTime():
    seed = 10086
    exp_name = "trainingTime"
    max_steps = 200000
    max_epoch = 1

    
    batch_size = 64
    per_gpu_batchsize = 64  # you should define this manually with per_gpu_batch_size=#

    
    # --------------------  调参  -----------------------
    matcher_coe = 0.5   # cost_sim_common + 0.5*cost_pick_common
    coe_loss = {"sim":1, "cls":1, "infoNCE":0.1}

    perceiver = True
    image_query_num = 64
    
    semantic_query_num = 70
    semantic_query_num_common = 50
    semantic_query_num_specific = 20
    
    # test_threshold_common = [0.4, 0.45, 0.5]
    # test_threshold_specific = [0.4, 0.45, 0.5]  

    # test_only = True
    # test_epoch = [41]