import os
import json
from tqdm import tqdm

dataset_name = "your dataset name"

def search_files(dir_path):
    """
    :param dir_path
    :return: list
    """
    result = []
    file_list = os.listdir(dir_path)  
    for file_name in file_list:
        complete_file_name = os.path.join(dir_path, file_name) 
        if os.path.isdir(complete_file_name): 
            result.extend(search_files(complete_file_name))
        if os.path.isfile(complete_file_name):  
            result.append(complete_file_name)
    return result


def read_and_write(files):
    json_file = r'../dataset/' + dataset_name + '.jsonl'
    with open(json_file, "a") as f:
        for i, file in enumerate(tqdm(files)):
            with open(file, 'r') as f1:
                content = f1.read()
            data = {
                "func": content,
                "idx": str(i),
            }
            json_data = json.dumps(data)
            f.write(json_data)
            f.write('\n')


def generate_label(files):
    vul = 0
    patch = 0
    txt_file = r'../dataset/' + dataset_name + '.txt'
    with open(txt_file, "a") as f:
        for i, file in enumerate(tqdm(files)):
            label = 0
            if file.find('VUL') != -1:
                label = 1
                vul = vul + 1
            elif file.find('vul') != -1:
                label = 1
                vul = vul + 1
            else:
                label = 0
                patch = patch + 1
            one_data = str(i)+'\t'+str(label)
            f.write(one_data)
            f.write('\n')
    print("vul num: ", vul)
    print("patch num: ", patch)


if __name__ == '__main__':
    dataset = search_files(r'./data/' + dataset_name)
    read_and_write(dataset)
    generate_label(dataset)
