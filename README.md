# Teaser


## Step 1: 数据预处理
1. 运行 `./preprocess/prePreprocess/clean_annotation.py`
<br/>

2. 运行 `./preprocess/generate_sentence_gallery.py`
<br>

3. 运行 `./preprocess/prePreprocess/cluster_sentences.py`

    > 1. 生成聚类模型 <br>
    > 
    > 2. 统计句子总频次和每个句子频次，根据出现频次及聚类结果，划分common数据集合specific数据集合
    
<br/>

4. 运行 `./preprocess/data_preprocess_dividually.py`

    > 分别处理train 和 val/test 数据


    <br>

5. 运行 `./preprocess/generate_CE_dividually.py`

    > 根据设定的common和specific数量，生成对应的generate_CE_dividually


## Step 2: Train & Evaluation
修改 `transq/config.py` 配置文件，运行 `run.py` 


## Step 3: 测试结果后处理

1. 在训练集上测试训练好的模型，用于记录query的平均位置
 

2. 运行 `evaluation/NLG_eval/test_from_json.py`，计算训练集中所提到的句子的平均位置顺序。对测试集结果进行重新排序，并获得最终指标。
    >  注意：由于IU X-ray和MIMIC-CXR之间存在细微差异（图像数量不同），我们为了方便起见，在两个独立的项目中实施了这两个数据集，而主分支是针对MIMIC-CXR的。



3. 测试CE指标

    > a. `./evaluation/CE_eval/generate_csv.py` 生成pred、target reports.csv文件
    >
    > b. 根据上一步生成的csv文件，通过chexpert预测14个类别的分类结果 
    > 
    > c. `./evaluation/CE_eval/calc_CE.py` 生成CE指标



