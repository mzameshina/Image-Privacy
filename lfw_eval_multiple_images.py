# Computes evaluation metrics (accuracyuracy, percentage of images in between closest original and modified images of the same person and recall) for image emedddings. With confounders. 


# LIBRARIES.
import torch
import torch.nn.functional as F
import os
import random
import argparse
import subprocess
import multiprocessing as mp
import json


def feat_list(features):
    if isinstance(features, list):
        features_n = features[0]
        for i in range(1, len(features)):
            features_n += features[i]
        features_n /= len(features)
        #return features_n
    return features
    
     

# Recall @ k, modified image as query
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

def first_nonzero(tens, axis=0):
    mask = (tens != 0)
    flag, idx =  ((mask.cumsum(axis) == 1) & mask).max(axis)
    if not flag:
         print("Identity uniformity failed")
    return idx.int().item()
    
# Identity uniformity.
def identity_uniformity_modified(number_images_per_person, table_output = False):
    iu = torch.zeros(1)
    for i in range(number_images_per_person):
        retrieved = (features_per_number[i] @ torch.cat(original_features_per_number[:i] + original_features_per_number[i+1:] + [confounders_features]).T).argsort(descending=True)
        
        for j in range(retrieved.size(0)):
            fn = first_nonzero(retrieved[j])
            iu += sum([(retrieved[j] == torch.ones(k)*i).any().float() ]) / retrieved.size(0)
            
    iu /= number_images_per_person
    iu = iu.float().mean()
    if table_output == False:
        print(f'Identity uniformity: (modified image as query): {100 * iu:.2f} %')
    return 100 * iu

    
if __name__ == "__main__":
    class MyFormatter(
        argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter,
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="magface",
        help="Embedding method for evaluation. Choose from: magface, facenet, sphereface, arcface, facexmobile, facexrn50",
        required=False,
    )
    parser.add_argument(
        "--number_images_per_person",
        type=int,
        default=5,
        help="Number of images present for each person",
        required=False,
    )
    parser.add_argument(
        "--features_directory",
        type=str,
        default="/private/home/mzameshina/FACE/FAWKES_VQGAN_lfw_5_only_FAWKES",
        help="File where features are located without '.pt' as well as a directory with images",
        required=False,
    )
    parser.add_argument(
        "--original_features_directory",
        type=str,
        default='/private/home/mzameshina/FACE/CC/experiments/casual_conversations_5_img_per_person_shape_456_456_3____',
        help="File where original image features are located without '.pt' as well as a directory with images",
        required=False,
    )
    parser.add_argument(
        "--confounder_features_directory",
        type=str,
        default='/private/home/mzameshina/FACE/UntitledF',
        help="File where confounder image features are located without '.pt' as well as a directory with images",
        required=False,
    )    
    parser.add_argument(
        "--k_values",
        type=int,
        nargs='+',
        default=[1, 2, 10, 100],
        help="k values for evaluating recall",
        required=False,
    )    
    parser.add_argument(
        "--table_output",
        type=bool,
        default=True,
        help="Choose to output a python table",
        required=False,
    )
    
    args = parser.parse_args()
    
    embedding_method = args.embedding
    number_images_per_person = args.number_images_per_person
    features_directory = args.features_directory
    original_features_directory = args.original_features_directory
    confounder_features_directory = args.confounder_features_directory
    k_values = args.k_values
    table_output = args.table_output
    
    # PARAMETERS.
    # Embedding method that was used for feature creation. Should be present at the end of the folder name. ('folder_name' + '_' + 'embedding_method').
    # embedding_method = 'magface'

    # Number of images per person, must be constant. 
    # number_images_per_person = 5

    # Feature directories.
    #features_directory = '/private/home/gcouairon/face/output/lfw5_kl16_facenet_emb0.03_rankhalf_1/images'
    #original_features_directory = 'lfw_5_images_per_person'
    #confounder_features_directory = 'data/lfw_crop_single'

    # StyleGAN features.
    #features_directory = '/private/home/marlenec/face_priv/stylegan2-ada-pytorch/out/projected_img_lfw_crop_filtered_5imgPerPerson_facenet/projected_img_lfw_crop_filtered_5imgPerPerson_facenet'

    # VQGAN features.
    #features_directory = '/private/home/mzameshina/FACE/lfw_kl16_facenet_emb0.03_rankhalf_1/images'

    # FAWKES features.
    #features_directory = '/private/home/mzameshina/FACE/lfw_5_images_per_person_FAWKES_only'

    # FAWKES + StyleGAN features.
    #features_directory = '/private/home/mzameshina/FACE/fawkesstylegan_projected_img_lfw_crop_filtered_5imgPerPerson_facenet'

    # FAWKES + VQGAN features.
    #features_directory = '/private/home/mzameshina/FACE/lfw_5_images_per_person_FAWKES_VQGAN__/'

    # StyleGAN + FAWKES features.
    #features_directory = '/private/home/mzameshina/FACE/FAWKES_StyleGAN_lfw_5_only_FAWKES'

    # VQGAN + FAWKES features.
    #features_directory = '/private/home/mzameshina/FACE/FAWKES_VQGAN_lfw_5_only_FAWKES'

    # Recall at k, k parameters
    # k_values = [1, 2, 10, 100]
    

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
    
    if not table_output:
        print(f'Percentage of images in between in database (modified image as query): {100 * percentage_modified:.2f} %')
        print(f'Percentage of images in between in database (original image as query): {100 * percentage_original:.2f} %')

        for k in k_values:
            recall_k_modified(int(k), number_images_per_person)

        for k in k_values:
            recall_k_original(int(k), number_images_per_person)
            
    else:
        data = {
            'Percentage: m.i.' : 100 * float(percentage_modified.numpy()),
            'Percentage: o.i.' : 100 * float(percentage_modified.numpy()),
           }
        
        for i in range(len(k_values)):
            k = k_values[i]
            name_recall_modified = "Recall@" + str(k) + ": m.i."
            name_recall_original = "Recall@" + str(k) + ": o.i."
            data[name_recall_modified] = float(recall_k_modified(int(k_values[i]), number_images_per_person, table_output).numpy())
            data[name_recall_original] = float(recall_k_original(int(k_values[i]), number_images_per_person, table_output).numpy())
        
        json_string = json.dumps(data)
        print(json_string)

        #rows = [["Proportion: m.i."], ["Proportion: o.i."]]
        #for k in k_values:
        #    rows.append(["Recall@" + str(k) + ": m.i."])
        #    rows.append(["Recall@" + str(k) + ": o.i."])
        #rows[0] += [100 * percentage_modified.numpy()]   
        #rows[1] += [100 * percentage_original.numpy()]
        #for i in range(len(k_values)):
        #    rows[i*2 + 2] += [recall_k_modified(int(k_values[i]), number_images_per_person, table_output).numpy()]
        #    rows[i*2 + 3] += [recall_k_original(int(k_values[i]), number_images_per_person, table_output).numpy()]
        #print(rows)