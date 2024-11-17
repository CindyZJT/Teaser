# env: v2update

import os
import copy

import random
import torch 
import numpy as np

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
# import lightning.pytorch.strategies
#from transq.modules.gallery import Gallery
from transq.config import ex
from transq.modules import TransformerSQ
# from transq.modules import TransformerSQ, TransformerSQ_CNN

from transq.datamodules.multitask_datamodule import MTDataModule
import warnings

from datetime import datetime

from thop import profile




@ex.automain
def main(_config):
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # torch.multiprocessing.set_start_method('spawn', force=True)
    _config = copy.deepcopy(_config)
    exp_name = f'{_config["exp_name"]}'

# TODO
    seed = _config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed)

    dm = MTDataModule(_config, dist=False)
    # dm = MTDataModule(_config, dist=True)
    
    tokenizer = dm.tokenizer
    vocab_size = dm.vocab_size
 
    if _config["test_only"]:
        pass
    elif _config["backbone"]=="ViT":
        model = TransformerSQ(_config, tokenizer)
        
        
    # elif _config["backbone"]=="CNN":
        # model = TransformerSQ_CNN(_config, tokenizer)
    #print(model)
    exp_name = f'{_config["exp_name"]}'


    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join("./_res_dir/checkpoints/", exp_name),
        filename = '{epoch}'+f'_seed{_config["seed"]}',
        save_top_k= -1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        every_n_epochs = 1,
        save_on_train_epoch_end = True,
        save_last=True,
    )
    

    
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    
    print("start trainer process...")
    
    trainer = pl.Trainer(
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="gpu",
        devices=_config["num_gpus"],
        strategy="auto",           
        # strategy="ddp",
        
        # benchmark=False,
        # deterministic=True,
        
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        # prepare_data_per_node=False,
        # replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        # flush_logs_every_n_steps=10,
        # resume_from_checkpoint=_config["resume_from"],
        # weights_summary="top",
        # max_depth = 0,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],   
        check_val_every_n_epoch= 1,  #n个epoch做一次验证？ 
        num_sanity_val_steps = 0,
        limit_val_batches = 0.1,    # 使用多少验证集做验证
        profiler="simple",
    ) 

    if not _config["test_only"]:
        print("seed : ", _config["seed"])
        trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])
        
    else:
        # _config["load_path"] = "./model/TranSQ-mimic.ckpt"
        
        # test_threshold = _config['test_threshold']        
        
        # for lamda in test_threshold:
        #     _config['test_threshold'] = lamda
        #     model = TransformerSQ(_config, tokenizer)
        #     trainer.test(model, datamodule=dm)
        #     # trainer.test(model, datamodule=dm, ckpt_path=_config["resume_from"])



# '''''''''                              
                    
        test_threshold_common = _config["test_threshold_common"]
        test_threshold_specific = _config['test_threshold_specific']        
        test_epoch = _config["test_epoch"]
        
     
        for epoch in test_epoch: 
            _config["load_path"] = "./_res_dir/checkpoints/{}/epoch={}_seed{}.ckpt".format(exp_name, epoch, _config["seed"])
            model = TransformerSQ(_config, tokenizer)
            model.epoch_count =  model.transformerSQHparams.config["load_path"].split("/")[-1].split("=")[1].split('_')[0].split('_')[0]
            
            for com_t in test_threshold_common: 
                _config['test_threshold_common'] = com_t
                model.test_threshold_common = _config['test_threshold_common']
                print('common threshold : ', model.test_threshold_common)
                
                for spe_t in test_threshold_specific:
                    _config['test_threshold_specific'] = spe_t
                    model.test_threshold_specific = _config['test_threshold_specific']
                    print('specific threshold : ', model.test_threshold_specific)

                    
                    trainer.test(model, datamodule=dm, ckpt_path=_config["resume_from"])

                    
                    

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(current_time)