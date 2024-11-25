import torch
from .tokenizer import Tokenizer
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from transformers import (
    # DataCollatorForLanguageModeling,
    # DataCollatorForWholeWordMask,
    BertTokenizer,
)


def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            BertTokenizer.from_pretrained(
                from_pretrained, do_lower_case="uncased" in from_pretrained
            )
        torch.distributed.barrier()
    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, _config):
        super().__init__()

        self.data_dir = _config["data_root"]
        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size
        self.data_name = _config["datasets"][0]
        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]
        self.max_sent_num = _config["max_sent_num"]
        self.sent_emb=_config["sent_emb"]
        self.draw_false_image = _config["draw_false_image"]
        self.draw_false_text = _config["draw_false_text"]
        self.pre_data_path = _config["pre_data_path"]
        self.image_only = _config["image_only"]

        self.train_transform_keys = (
            ["default_train"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )

        self.val_transform_keys = (
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
        )

        #tokenizer = _config["tokenizer"]
        #self.tokenizer = get_pretrained_tokenizer(tokenizer)
        #self.vocab_size = self.tokenizer.vocab_size
        self.threshold = 3
        self.tokenizer = Tokenizer(self.data_dir, self.threshold, self.data_name)
        self.vocab_size = self.tokenizer.get_vocab_size()
        print("vocab_size",self.vocab_size)
        #collator = (
        #    DataCollatorForWholeWordMask
        #    if _config["whole_word_masking"]
        #    else DataCollatorForLanguageModeling
        #)

        #self.mlm_collator = collator(
        #    tokenizer=self.tokenizer, mlm=True, mlm_probability=_config["mlm_prob"]
        #)
        self.setup_flag = False

    #@property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            self.train_transform_keys,
            #threshold = 3,
            split="train",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            max_sent_num=self.max_sent_num,
            sent_emb=self.sent_emb,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            pre_data_path = self.pre_data_path,
            image_only=self.image_only,
            #data_name = self.data_name
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            #threshold = 3,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            max_sent_num=self.max_sent_num,
            sent_emb=self.sent_emb,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            pre_data_path = self.pre_data_path,
            image_only=self.image_only,
            #data_name = self.data_name
        )

        if hasattr(self, "dataset_cls_no_false"):
            self.val_dataset_no_false = self.dataset_cls_no_false(
                self.data_dir,
                self.val_transform_keys,
                split="val",
                #threshold = 3,
                image_size=self.image_size,
                max_text_len=self.max_text_len,
                max_sent_num=self.max_sent_num,
                sent_emb=self.sent_emb,
                draw_false_image=0,
                draw_false_text=0,
                image_only=self.image_only,
                #data_name = self.data_name
            )

    def make_no_false_val_dset(self, image_only=False):
        return self.dataset_cls_no_false(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            #threshold = 3,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            max_sent_num=self.max_sent_num,
            sent_emb=self.sent_emb,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,
            #data_name = self.data_name
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            #threshold=3,
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            max_sent_num=self.max_sent_num,
            sent_emb=self.sent_emb,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            pre_data_path = self.pre_data_path,
            image_only=self.image_only,
            #data_name=self.data_name
        )

    def setup(self, stage):
        if not self.setup_flag:
            #print("tokenizer")
            #print(self.tokenizer)
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.train_dataset.tokenizer = self.tokenizer
            self.val_dataset.tokenizer = self.tokenizer
            self.test_dataset.tokenizer = self.tokenizer

            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,

        )
        return loader
