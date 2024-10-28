import os, sys
import csv
import json
import argparse
from glob import glob

from numpy import argmax, stack
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support

import torch
import torch.nn.functional as F
import datasets


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=False, default="./metrics_gen")

    args = parser.parse_args()
    return args

def cleaning_generation(input):
    return input.strip()

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def change_data_format(gen_data):
    transformed_data = {}
    for task, task_data in gen_data.items():
        transformed_data[task] = {}
        for config in task_data['preds'].keys():
            transformed_data[task][config] = {
                'preds': task_data['preds'][config],
                'golds': task_data['golds'][config]
            }
    return transformed_data

def get_metrics(restructured_data, model_name, output_dir):
    metrics = []
    for task, config_data in restructured_data.items():
        for config_name, data in config_data.items():
            if data['preds'] != [] and data['golds'] != []:
                print(f"===========Task: {task}, config: {config_name}============")
                preds = [i.strip() for i in data['preds']]
                golds = [i.strip() for i in data['golds']]
                cls_report = classification_report(golds, preds, output_dict=True)
                micro_f1, micro_prec, micro_rec, _ = precision_recall_fscore_support(golds, preds, average='macro')
                print("accuracy: ", cls_report['accuracy'])
                print("micro f1: ", micro_f1)
                print("macro f1: ", cls_report['macro avg']['f1-score'])
                print("weighted f1: ", cls_report['weighted avg']['f1-score'])
                print("======\n\n")
                metrics.append({
                    'task': task,
                    'config': config_name,
                    'accuracy': cls_report['accuracy'],
                    "micro_prec": micro_prec,
                    "micro_rec": micro_rec,
                    'micro_f1_score': micro_f1,
                    "macro_prec": cls_report['macro avg']['precision'],
                    "macro_rec": cls_report['macro avg']['recall'],
                    'macro_f1_score': cls_report['macro avg']['f1-score'],
                    "weighted_prec": cls_report['weighted avg']['precision'],
                    "weighted_rec": cls_report['weighted avg']['recall'],
                    'weighted_f1': cls_report['weighted avg']['f1-score'],
                })
            else:
                print(f"Task: {task}, config: {config_name} has no data")
                metrics.append({
                    'task': task,
                    'config': config_name,
                    'accuracy': "",
                    "micro_prec": "",
                    "micro_rec": "",
                    'micro_f1_score': "",
                    "macro_prec": "",
                    "macro_rec": "",
                    'macro_f1_score': "",
                    "weighted_prec": "",
                    "weighted_rec": "",
                    'weighted_f1': "",
                })
    pd.DataFrame(metrics).reset_index().to_csv(f'{output_dir}/results_{model_name.split("/")[-1]}.csv', index=False)
    

def gen_results_overview(gen_data):
    for task, config_data in gen_data.items():
        print(f"Task: {task}")
        for config_name, data in config_data.items():
            print(f"========config: {config_name}=========")
            print("preds")
            print(set(data['preds']))
            print("golds")
            print(set(data['golds']))

if __name__ == '__main__':
    args = parse_arguments()
    gen_data = read_json(args.results_file)
    model_name = list(gen_data.keys())[0]
    gen_data = gen_data[model_name]
    # print(gen_data)
    print("Evaluating model: ", model_name)
    get_metrics(change_data_format(gen_data), model_name, args.output_dir)





