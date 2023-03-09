from tabulate import tabulate
from texttable import Texttable
import latextable
import random
import json
from os.path import exists
import os
import argparse
import itertools
import math
from PIL import Image
import glob
import subprocess
import multiprocessing as mp
import sys
from types import SimpleNamespace as nspace
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import torch
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
from functools import partial
import torch.nn.functional as F


# Embedding methods used in evaluation.
embeddings = {}

embeddings["facenet"] = "FaceNet"
embeddings["magface"] = "MagFace"
embeddings["sphereface"] =  "SphereFace"
embeddings["arcface"] = "ArcFace"
embeddings["facexmobile"] = "FaceXMobile"
embeddings["facexrn50"] = "FaceXRN50"
k_values = [1,3,5,10,50,100]

#run linux command
def run_command(command, logs = '/private/home/mzameshina/FACE/exp_results_image_privacy/r.log'):
    print(command)
    if logs:
        with open(logs, "a") as text_file:
            text_file.write(str(command + '\n'))
    out = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, bufsize=0)
    for line in iter(out.stdout.readline, b''):
        print ('>>> {}'.format(line.rstrip()))
        if logs:
            with open(logs, "a") as text_file:
                text_file.write(str(line) + str('\n'))
    for line in iter(out.stderr.readline, b''):
        print ('>>> {}'.format(line.rstrip()))
        if logs:
            with open(logs, "a") as text_file:
                text_file.write(str(line) + str('\n'))


def feat_list(features):
    if isinstance(features, list):
        features_n = features[0]
        for i in range(1, len(features)):
            features_n += features[i]
        features_n /= len(features)
    return features


def json_gen(original_features_directory, confounder_features_directory, features_directory, embedding_method, number_images_per_person):
    # LOAD FEATURES.
    # Features to test.
    features = torch.load(f'{features_directory}_{embedding_method}.pt')
    features = feat_list(features)
    features = F.normalize(features)
    features_per_number = []
    for i in range(number_images_per_person):
        features_per_number += [features[i::number_images_per_person]]

    # Features of original images.
    original_features = torch.load(f'{original_features_directory}_{embedding_method}.pt')
    original_features = feat_list(original_features)
    original_features = F.normalize(original_features)

    original_features_per_number = []
    for i in range(number_images_per_person):
        original_features_per_number += [original_features[i::number_images_per_person]]

    # Confounders.
    confounders_features = torch.load(f'{confounder_features_directory}_{embedding_method}.pt')
    confounders_features = feat_list(confounders_features)
    confounders_features = F.normalize(confounders_features)

    confounders_features_per_number = []
    for i in range(number_images_per_person):
        confounders_features_per_number += [confounders_features[i::number_images_per_person]]
    
    # Tensor of image identities: arange_array[$image_A_number$] = $image_A_identity$.
    arange_list = []
    for i in range(len(original_features)):
        arange_list += [i for j in range(number_images_per_person)]
    arange_array = torch.tensor(arange_list)    

    arange_list = []
    for i in range(len(original_features)):
        arange_list += [i for j in range(number_images_per_person-1)]
    arange_array = torch.tensor(arange_list)    
    
    
    # EVALUATION METRICS.
    # Dataset percentage of images in between (average rank): modified image as query.
    percentage_modified = torch.zeros(1)
    for i in range(number_images_per_person):
        retrieved = (features_per_number[i] @ torch.cat(original_features_per_number[:i] + original_features_per_number[i+1:] + [confounders_features]).T).argsort(descending=True)
        percentage_modified += sum([(retrieved[i] == i).nonzero().item() for i in range(retrieved.size(0))]) / (retrieved.size(0) * retrieved.size(1)) 
    percentage_modified /= number_images_per_person
    percentage_modified = percentage_modified.float().mean()
    
    # Dataset percentage of images in between (average rank); original image as query.
    percentage_original = torch.zeros(1)
    for i in range(number_images_per_person):
        retrieved = (original_features_per_number[i] @ torch.cat(features_per_number[:i] + features_per_number[i+1:] + [confounders_features]).T).argsort(descending=True)
        percentage_original += sum([(retrieved[i] == i).nonzero().item() for i in range(retrieved.size(0))])/(retrieved.size(0) * retrieved.size(1)) 
    percentage_original /= number_images_per_person
    percentage_original = percentage_original.float().mean()
    
    data = {
            'Percentage: m.i.' : 100 * float(percentage_modified.numpy()),
            'Percentage: o.i.' : 100 * float(percentage_modified.numpy()),
           }
        
    def recall_k_modified(k, number_images_per_person, table_output = False):
        recall = torch.zeros(1)
        for i in range(number_images_per_person):
            retrieved = (features_per_number[i] @ torch.cat(original_features_per_number[:i] + original_features_per_number[i+1:] + [confounders_features]).T).argsort(descending=True)
            recall += sum([(retrieved[i][:k] == torch.ones(k)*i).any().float() for i in range(retrieved.size(0))]) / retrieved.size(0)
        recall /= number_images_per_person
        recall = recall.float().mean()
        if table_output == False:
            print(f'Recall at {k}: (modified image as query): {100 * recall:.2f} %')
        return 100 * recall


    # Recall @ k, original image as query
    def recall_k_original(k, number_images_per_person, table_output = False):
        recall = torch.zeros(1)
        for i in range(number_images_per_person):
            retrieved = (original_features_per_number[i] @ torch.cat(features_per_number[:i] + features_per_number[i+1:] + [confounders_features]).T).argsort(descending=True)
            recall += sum([((retrieved[i][:k] == torch.ones(k)*i).any().float()) for i in range(retrieved.size(0))]) / retrieved.size(0)
        recall /= number_images_per_person
        recall = recall.float().mean()
        if table_output == False:
            print(f'Recall at {k}: (original image as query): {100 * recall:.2f} %')
        return 100 * recall

    for i in range(len(k_values)):
        k = k_values[i]
        name_recall_modified = "Recall@" + str(k) + ": m.i."
        name_recall_original = "Recall@" + str(k) + ": o.i."
        data[name_recall_modified] = float(recall_k_modified(int(k_values[i]), number_images_per_person).numpy())
        data[name_recall_original] = float(recall_k_original(int(k_values[i]), number_images_per_person).numpy())
        
    return data
        


#PARAMETERS
os.environ['CUDA_VISIBLE_DEVICES']='2, 3'



# Function for printing tables into LaTex
# Input :
# rows -- table in list format
# caption -- caption of a table, optional
def latex_table(rows, caption=""):
    #print(rows)
    table = Texttable()
    table.set_cols_align(["c"] * len(rows[0]))
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)
    t_print = latextable.draw_latex(table, caption)
    return(t_print)

def emb(folder, m):
    run_command('python3 embed.py -m ' + m + ' -f ' + folder)
    
# Runs embedding generation for a given folder
def run_emb_generation(embedding_method1, folder, embedding_method2 = None, restart = False):
    emb(folder, embedding_method1)
        
    if embedding_method2 != None:
        emb(folder, embedding_method2)


# Runs evaluation for a given image folder and an embedding method.
# More information is in lfw_eval_multiple_images.py
def run_evaluation(embedding_method, number_images_per_person, features_dir, 
                   original_features_dir, confounder_features_dir, k_values, table_output = True, output_to_file = True):
    print(embedding_method)
    k_val_string = ""
    output_file = "/private/home/mzameshina/FACE/tables/" + embedding_method + "_" + str(number_images_per_person) + features_dir.split("/")[-1] + "_" + str(random.randint(1000000000000, 10000000000000-1)) + ".txt"
    for k in k_values:
        k_val_string += str(k) + " "
    of = "> " + output_file
    if output_to_file == False:
        of = ""
    run_command('python3 /private/home/mzameshina/FACE/lfw_eval_multiple_images.py --embedding ' + str(embedding_method) + ' --number_images_per_person ' + str(number_images_per_person) + ' --features_dir ' + str(features_dir) + ' --original_features_dir ' + str(original_features_dir) + ' --confounder_features_dir '+ str(confounder_features_dir)  + ' --k_values ' + str(k_val_string) + ' --table_output ' + " " + str(of))
    return output_file

# Creates a table for printing from multiple json files.
# Input : 
# files -- list of json files of evaluation results for a giv en dataset.
# datasets -- list of datasets that have been evaluated.
def json_to_rows(data_list, datasets):
    
    rows = [[" "] + datasets]
    
    for k in data_list[0].keys():
        rows.append([k])

    for i in range(len(data_list)):
        data = data_list[i]
        for i in range(0, len(data.keys())):
            k = rows[i+1][0]
            rows[i + 1].append(data[k])
    return rows


def create_evaluation_experiment(embedding, datasets, dataset_names, original_dir, confounder_dir, number_images_per_person, k_values):
    #run_emb_generation(embedding, original_dir)
    #run_emb_generation(embedding, confounder_dir)
    evaluation_files = []
    for d in datasets:
        #run_emb_generation(embedding, d)
        output_file = run_evaluation(embedding, number_images_per_person, d, original_dir, confounder_dir, k_values)
        evaluation_files.append(output_file)
    
    data = []
    for i in range(len(evaluation_files)):
        data.append(json_gen(original_dir, confounder_dir, datasets[i], embedding, number_images_per_person))
    
    rows = json_to_rows(data, dataset_names)
    return evaluation_files, rows


# Compute average metric value (calc_param) for all the embedding methods, excluding exceptions (exceptions) for the given experiment (experiment_name).
def average_recall(results, exceptions, experiment_name, calc_param = ['Recall@10: o.i.', 'Recall@10: m.i.']):
    sum_recall = 0
    num = 0
    
    ind = -1
    names = sorted([v for v in results.values()])[0][0]
    
    #print(names)
    
    for i in range(len(names)):
        if experiment_name == names[i]:
            ind = i
    
    if ind == -1:
        #print('Experiment name is not found')
        #print("Experiment names:", names)
        return 0
            
    
    for k in results.keys():
        if k not in exceptions:
            v = results[k]
            for l in v:
                if l[0] in calc_param:
                    num += 1
                    sum_recall +=  l[ind]
                    
    if num == 0:
        #print('No embedding method or calculation parameter fits the requirement')
        return 0

    else:
        return sum_recall / num
    
    
def average_recall_comparisson(results, exceptions, folder_names=None):
    if folder_names == None:
        folder_names = sorted([v for v in results.values()])[0][0][1:]

    dic = {}
    for ex in folder_names:
        dic[ex] = average_recall(results, exceptions, ex)
    return dic


def resize(folder_list, tag = '_resized_112_112', k = 100000):
    new_folders=[]
    for folder in folder_list:
        l = 0
        path = folder + tag
        if not os.path.exists(path):
            os.makedirs(path)
        for filename in sorted(os.listdir(folder)): #path of raw images
            if l < k and filename.find('.jpg')*filename.find('.jpeg')*filename.find('.png') != -1:
                img = Image.open(folder + '/' + filename).resize((112,112))
                img.save('{}{}{}'.format(path,'/',os.path.split(filename)[1]))
            l += 1
        new_folders.append(path)
    return new_folders


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dir", help="Directory with original (unmodified images)", required=False, default="/private/home/mzameshina/FACE/lfw_5_images_per_person")
    parser.add_argument("--confounder_features_dir", help="Directory with confounders", required=False, default="/private/home/mzameshina/FACE/data/lfw_crop_single")
    parser.add_argument("--folder_paths", help="All the folders to be transformed to private", required=True)
    parser.add_argument("--folder_names", help="Names of each folder from folder_paths", required=True)
    parser.add_argument("--results_folder", help="Directory to save results to", required=False, default='/private/home/mzameshina/FACE/exp_results_image_privacy/')
    parser.add_argument("--num_images_per_person", help="Number of images per each person present in the datasets", required=False, default='5')
    parser.add_argument("--experiment_name", help="Current experiments title", required=False, default='lfw-experiment')
    parser.add_argument('-m', '--method', type=str, default='magface', help='face embedding method to use', required=False)
    parser.add_argument('-m1', '--method_1', type=str, default='facexmobile', help='face embedding method to use', required=False)
    args = parser.parse_args()    

    original_dir = args.original_dir
    confounder_features_dir = args.confounder_features_dir
    folder_paths = args.folder_paths
    folder_names = args.folder_names
    results_folder = args.results_folder
    num_images_per_person = int(args.num_images_per_person)
    experiment_name = args.experiment_name
    method = args.method
    method_1 =  args.method_1

    
    methods = [method, method_1]
    if method_1 == None:
        methods = [method]
    if method == 'all':
        methods = class_map.keys()
    emb_exclude = methods


    def print_(*msg, fd = results_folder + experiment_name + '.log'):
        '''print and log!'''
        import datetime as dt
        message = []
        for m in msg:
            message.append(str(m))
        message = ' '.join(message)
        with open(fd,'a') as log:
            log.write(f'{dt.datetime.now()} | {message}\n')
        
    def evaluation_experiment_all_embeddings(embeddings, features_dir, dataset_names, original_features_dir, confounder_features_dir, number_images_per_person, k_values, caption):
        results = {}
    
        for e in embeddings.keys():
            evaluation_files_, rows_ = create_evaluation_experiment(e, features_dir, dataset_names, original_features_dir, confounder_features_dir, number_images_per_person, k_values)
            print_(latex_table(rows_, caption=caption + "and tested with " + embeddings[e]))
            results[e] = rows_
        return results


    comp_dirs_resized = resize([original_dir, confounder_features_dir])
    exp_dirs_resized = resize(folder_paths)

    for e in embeddings.keys():
      for f in comp_dirs_resized + exp_dirs_resized + folder_paths + [original_dir, confounder_features_dir]:
          run_emb_generation(e, f, restart=False)  
  

    results = evaluation_experiment_all_embeddings(embeddings, exp_dirs_resized, folder_names, comp_dirs_resized[0], comp_dirs_resized[1], num_images_per_person, k_values, caption=experiment_name) 


    c = average_recall_comparisson(results, emb_exclude)
    
    for k in sorted(list(c.keys())):
      print_("\item " + str(k) + " "  + str(c[k]))