import os
import json
import mmcv
import numpy as np
import sys
import pickle as pkl

sys.path.append("..")
sys.path.append("/home/zhaojunting/sourceCode/updateTranSQ/v8Specific")





# specific 和common库
specific_gallery_path = "/home/zhaojunting/sourceCode/updateTranSQ/v8Specific/preprocess/data/specific_sentence_gallery.pkl"

f2 = open(specific_gallery_path, "rb")
specific_gallery = pkl.load(f2)
specific_sentence_gallery = specific_gallery.sentence_gallery 
