# PVDetector: Pretrained Vulnerability Detection on Vulnerability-enriched Code Semantic Graph [TOSEM 2025]
## :rocket:Highlight
PVDetector is a novel approach that utilizes rich code semantics for precise vulnerability detection. At its core, PVDetector employs a new model called Vulnerability-enriched Code Semantic Graph (VCSG), which accurately characterizes functions by distinguishing the semantics of identical variables and more finely capturing control dependencies, data dependencies, and vulnerability relationships.

![image](https://github.com/yoimiya-nlp/PVDetector/blob/main/PVDetector.png)
## :wrench:How to Use
### Step 0: Dataset Preparation
There are two ways to build a dataset. 

**Way 1: Build from source files.** You need to run the ```generate_json_and_txt_from_source_file.py``` file in the ```data``` folder to generate two files: ```dataset.jsonl``` and ```dataset.txt```. Then, use the ```Preprocess``` class on line 358 of the ```data_preprocess.py``` file for preprocessing.
```
cd data
python generate_json_and_txt_from_source_file.py
```
**Way 2: Build from a JSONL file containing labels.** Use the ```Preprocess_JSONL``` class on line 359 of the ```data_preprocess.py``` file for preprocessing.
### Step 1: Preprocessing
Preprocess the dataset ```dataset.jsonl``` and split it into ```dataset_train.pkl``` and ```dataset_test.pkl```.
```
sh preprocess.sh
```
This process will generate the VCSG and place the preprocessed dataset in the ```preprocessed_data``` folder.
### Step 2: Training
**Train the PVDetector model**, remember to set ```DATASET``` to the dataset name you want. Meanwhile, ```FROM_CHECKPOINT``` and ```TO_CHECKPOINT``` respectively indicate the weight from which training starts and the location where the weight is stored after training.
```
sh train.sh
```
### Step 3: Testing
**Test the PVDetector model**, use ```DATASET``` and ```FROM_CHECKPOINT``` to choose the dataset and weights, thereby testing the vulnerability detection performance of PVDetector.
```
sh test.sh
```
## :star:Citation
If you find this repo useful, please cite our paper.
```
comming soon
```
