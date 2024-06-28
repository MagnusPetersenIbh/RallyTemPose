import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat
import os 
import argparse
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
#from TemPose_gcn_trans import Transformer

from baselineModels import Transformer
#from TemPose_gcn_trans import Transformer
from utils import *
from Utils.tools import *
import optuna
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


def custom_train_test_splitID(grouped_data, encoded_labels,grouped_position=None,grouped_ID = None, test_size=0.16, random_state=None):
    """
    Custom train-test split for grouped data and labels.

    Parameters:
    grouped_data (list of lists): Grouped input data.
    encoded_labels (list of lists): Grouped encoded labels.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
    tuple: Split training and test sets for both data and labels.
    """
    # Generate indices for rallies
    rally_indices = list(range(len(grouped_data)))

    # Split rally indices ### split videos here
    train_indices, test_indices = train_test_split(rally_indices, test_size=test_size, random_state=random_state)

    # Function to subset data based on indices
    subset_data = lambda data, indices: [data[i] for i in indices]

    # Create training and test sets
    train_data = subset_data(grouped_data, train_indices)
    test_data = subset_data(grouped_data, test_indices)
    train_labels = subset_data(encoded_labels, train_indices)
    test_labels = subset_data(encoded_labels, test_indices)
    if (grouped_position is not None) and (grouped_ID is not None):
        train_position = subset_data(grouped_position, train_indices)
        test_position = subset_data(grouped_position, test_indices)
        train_ID = subset_data(grouped_ID, train_indices)
        test_ID = subset_data(grouped_ID, test_indices)
        return train_data, test_data, train_labels, test_labels, train_position, test_position, train_ID, test_ID
    


    return train_data, test_data, train_labels, test_labels

def pad_list_of_arraysID(dataset, labels,position=None,playerID=None, padding_value=0,max_len=None):
    """
    Pad elements of the dataset where each element is a list of arrays.
    Each array has shape (N, T, P, D) and T can vary.

    Args:
    dataset (list of list of np.array): Dataset with elements as lists of arrays.
    labels (list of np.array): List of 1D label arrays with variable length.
    padding_value (int or float): Value used for padding. Defaults to 0.

    Returns:
    (list of list of np.array, list of np.array): Tuple of padded dataset and labels.
    """
    # Determine the maximum sequence length in the dataset
    if max_len is None:
        max_seq_len = max(len(element) for element in dataset)
    else:
        max_seq_len = max_len

    # Pad each element in the dataset and record original sequence lengths
    padded_dataset = []
    original_lengths = []  # To store the original sequence lengths
    if position is None:
        for element in dataset:
            original_length = len(element)
            original_lengths.append(original_length)

            padding_needed = max_seq_len - original_length
            if padding_needed > 0:
                # Assuming you can determine N, P, D from the existing arrays
                # and choosing an appropriate value for T
                N, P = element[0].shape[0], element[0].shape[2]
                T = np.min([arr.shape[1] for arr in element])  # Or use another method to determine T
                padding_array = np.full((N, T, P), 0.0)
                padded_element = element + [padding_array] * padding_needed
            else:
                padded_element = element
            padded_dataset.append(padded_element)
    else:
        padded_position = []
        padded_IDs = []
          # To store the original sequence lengths
        for element,pos,ID in zip(dataset,position,playerID):
            original_length = len(element)
            original_lengths.append(original_length)

            padding_needed = max_seq_len - original_length
            if padding_needed > 0:
                # Assuming you can determine N, P, D from the existing arrays
                # and choosing an appropriate value for T
                
                N, P = element[0].shape[0], element[0].shape[2]
                N, P_id = ID[0].shape[0],ID[0].shape[-1]
                N, P_dim = pos[0].shape[1], pos[0].shape[2]
                T = np.min([arr.shape[1] for arr in element])  # Or use another method to determine T
                padding_array = np.full((N, T, P), 0.0)
                padding_pos = np.full((T, N, P_dim), 0.0)
                padding_ID = np.full((N,P_id),int(0))
                padded_element = element + [padding_array] * padding_needed
                padded_pos = pos + [padding_pos] * padding_needed
                padded_ID = ID + [padding_ID] * padding_needed
            else:
                padded_element = element
                padded_pos = pos
                padding_ID = ID
            padded_dataset.append(padded_element)
            padded_position.append(padded_pos)
            padded_IDs.append(padded_ID)

    # Pad each array in the labels
    padded_labels = []
    for label in labels:
        padding_needed = max_seq_len - len(label)
        padded_label = np.pad(label, (0, padding_needed), 'constant', constant_values=padding_value)
        padded_labels.append(padded_label)

    return padded_dataset, padded_labels, original_lengths,padded_position,padded_IDs

def filter_sequences_by_lengthID(dataset, labels, positions = None, playerIDs = None , min_length=3, max_length=40):
    """
    Filter the dataset and labels to keep only sequences with lengths within
    the specified range.

    Args:
    dataset (list of list of np.array): Dataset with elements as lists of arrays.
    labels (list of np.array): Corresponding labels.
    min_length (int): Minimum acceptable sequence length.
    max_length (int): Maximum acceptable sequence length.

    Returns:
    (list of list of np.array, list of np.array): Filtered dataset and labels.
    """
    filtered_dataset = []
    filtered_labels = []
    filtered_position = []
    filtered_playerID = []
    

    for element, label, position,ID in zip(dataset, labels, positions, playerIDs):
        seq_length = len(element)
        if min_length <= seq_length <= max_length:
            filtered_dataset.append(element)
            filtered_labels.append(label)
            filtered_position.append(position)
            filtered_playerID.append(ID)

    return filtered_dataset, filtered_labels, filtered_position,filtered_playerID



class player_encoding_table:
    def __init__(self,match_info):
        strings = np.unique(np.concatenate([match_info['winner'],match_info['loser']]))
        print(f'Players in the dataset {strings}')
        self.player_encoding_table = {string: i for i, string in enumerate(strings)} ## plus 1 to make room for padding # no padding at the moment
        print("Encoding table:",self.player_encoding_table)
    def encode_string(self, s):
        return self.player_encoding_table.get(s, None)  # Returns None if the string is not found


def get_player_id(l,match_info,pet):
    if not match_info[match_info['video'] == l[3]]['downcourt'].values[0]:
        if l[4] == 'set1':
            if l[5]=='A':
                return [pet.encode_string(match_info[match_info['video'] == l[3]]['winner'].values[0]),1],[pet.encode_string(match_info[match_info['video'] == l[3]]['loser'].values[0]),0]
            elif l[5]=='B':
                return [pet.encode_string(match_info[match_info['video'] == l[3]]['winner'].values[0]),0],[pet.encode_string(match_info[match_info['video'] == l[3]]['loser'].values[0]),1]
            else:
                print('something went wrong with player reading')
                return -1
        elif l[4] == 'set2':
            if l[5]=='A':
                return [pet.encode_string(match_info[match_info['video'] == l[3]]['loser'].values[0]),0],[pet.encode_string(match_info[match_info['video'] == l[3]]['winner'].values[0]),1]
                
            elif l[5]=='B':
                return [pet.encode_string(match_info[match_info['video'] == l[3]]['loser'].values[0]),1],[pet.encode_string(match_info[match_info['video'] == l[3]]['winner'].values[0]),0]
                
            else:
                print('something went wrong with player reading')
                return -1
        elif l[4] == 'set3':
            if l[5]=='A':
                return [pet.encode_string(match_info[match_info['video'] == l[3]]['winner'].values[0]),1],[pet.encode_string(match_info[match_info['video'] == l[3]]['loser'].values[0]),0]
            elif l[5]=='B':
                return [pet.encode_string(match_info[match_info['video'] == l[3]]['winner'].values[0]),0],[pet.encode_string(match_info[match_info['video'] == l[3]]['loser'].values[0]),1]
            else:
                print('something went wrong with player reading')
                return -1
        else:
            print('set reading got weird input')

    elif match_info[match_info['video'] == l[3]]['downcourt'].values[0]:
        if l[4] == 'set1':
            if l[5]=='A':
                return [pet.encode_string(match_info[match_info['video'] == l[3]]['loser'].values[0]),0],[pet.encode_string(match_info[match_info['video'] == l[3]]['winner'].values[0]),1]     
            elif l[5]=='B':
                return [pet.encode_string(match_info[match_info['video'] == l[3]]['loser'].values[0]),1],[pet.encode_string(match_info[match_info['video'] == l[3]]['winner'].values[0]),0]
            else:
                print('something went wrong with player reading')
                return -1
        elif l[4] == 'set2':
            if l[5]=='A':
                return [pet.encode_string(match_info[match_info['video'] == l[3]]['winner'].values[0]),1],[pet.encode_string(match_info[match_info['video'] == l[3]]['loser'].values[0]),0]
            elif l[5]=='B':
                return [pet.encode_string(match_info[match_info['video'] == l[3]]['winner'].values[0]),0],[pet.encode_string(match_info[match_info['video'] == l[3]]['loser'].values[0]),1]
            else:
                print('something went wrong with player reading')
                return -1
        elif l[4] == 'set3':
            if l[5]=='A':
                return [pet.encode_string(match_info[match_info['video'] == l[3]]['loser'].values[0]),0],[pet.encode_string(match_info[match_info['video'] == l[3]]['winner'].values[0]),1]     
            elif l[5]=='B':
                return [pet.encode_string(match_info[match_info['video'] == l[3]]['loser'].values[0]),1],[pet.encode_string(match_info[match_info['video'] == l[3]]['winner'].values[0]),0]
            else:
                print('something went wrong with player reading')
                return -1
        else:
            print('set reading got weird input')
    
    else: 
        print('mistake in reading downcourt bool')
        return -1


def convert_AlphaOpenposeCoco_to_standard16Joint(pose_x):
    """
    pose_x: nx17x2
    taken from
    https://zhuanlan.zhihu.com/p/367707179
    """
    hip = 0.5 * (pose_x[:, 11] + pose_x[:, 12])
    neck = 0.5 * (pose_x[:, 5] + pose_x[:, 6])
    spine = 0.5 * (neck + hip)

    # head = 0.5 * (pose_x[:, 1] + pose_x[:, 2])

    head_0 = pose_x[:, 0]  # by noise
    head_1 = (neck - hip)*0.5 + neck  # by backbone
    head_2 = 0.5 * (pose_x[:, 1] + pose_x[:, 2])  # by two eye
    head_3 = 0.5 * (pose_x[:, 3] + pose_x[:, 4])  # by two ear
    head = head_0 * 0.1 + head_1 * 0.6 + head_2 * 0.1 + head_3 * 0.2

    combine = np.stack([hip, spine, neck, head])  # 0 1 2 3 ---> 17, 18, 19 ,20
    combine = np.transpose(combine, (1, 0, 2))
    combine = np.concatenate([pose_x, combine], axis=1)
    reorder = [17, 12, 14, 16, 11, 13, 15, 18, 19, 20, 5, 7, 9, 6, 8, 10]
    standart_16joint = combine[:, reorder]
    return standart_16joint

import scipy
def keypoint_smoothing(keypoints):
    x = keypoints.copy()

    if len(x.shape)>3:
        A,B,C,D = x.shape
        x = x.reshape(-1,C,D)
        window_length = 5
        polyorder = 2
        out = scipy.signal.savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=0, mode='interp', cval=0.0)
        return out.reshape(A,B,C,D)
    else:
        window_length = 5
        polyorder = 2
        out = scipy.signal.savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=0, mode='interp', cval=0.0)
        return out

def stack_grouped_data_into_lists(grouped_data, grouped_labels, grouped_pos, grouped_player_id):
    """
    Organizes grouped data into separate lists for data, labels, positions, and player IDs,
    segmented by match, while maintaining the original (match, set, rally) groupings.

    Parameters:
    - grouped_data (dict): Dictionary with keys as (match_name, set_name, rally_number) tuples and grouped input data as values.
    - grouped_labels (dict): Similar structure as grouped_data but contains encoded labels.
    - grouped_pos (dict): Similar structure as grouped_data but contains position data.
    - grouped_player_id (dict): Similar structure as grouped_data but contains player ID data.

    Returns:
    dict: A dictionary where keys are match names and values are dictionaries. Each dictionary contains lists for 'data', 'labels', 'pos', and 'player_id', keeping the original groupings.
    """
    organized_by_match = {}

    for key in grouped_data:
        match_name = key[0]  # Extract match name from key
        
        if match_name not in organized_by_match:
            organized_by_match[match_name] = {'data': [], 'labels': [], 'pos': [], 'player_id': []}

        # Append data for this group to the lists under the correct match name
        organized_by_match[match_name]['data'].append(grouped_data[key])
        organized_by_match[match_name]['labels'].append(grouped_labels[key])
        organized_by_match[match_name]['pos'].append(grouped_pos[key])
        organized_by_match[match_name]['player_id'].append(grouped_player_id[key])

    return organized_by_match

import numpy as np
from sklearn.model_selection import KFold
def custom_k_fold_cross_validation(grouped_data, encoded_labels,grouped_position,grouped_playerID, k=5, random_state=None, shuffle=True):
    """
    Custom k-fold cross-validation for grouped data and labels, handling jagged lists.

    Parameters:
    grouped_data (list of lists): Grouped input data.
    encoded_labels (list of lists): Grouped encoded labels.
    k (int): Number of folds.
    random_state (int): Controls the random shuffling of the data.
    shuffle (bool): Whether to shuffle data before splitting into batches.

    Yields:
    generator of tuples: Each iteration returns training and validation sets for both data and labels.
    """
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)

    indices = list(range(len(grouped_data)))  # Use simple range list for indexing

    for train_indices, val_indices in kf.split(indices):
        # Manual indexing function for jagged lists
        train_data = [grouped_data[i] for i in train_indices]
        val_data = [grouped_data[i] for i in val_indices]
        train_labels = [encoded_labels[i] for i in train_indices]
        val_labels = [encoded_labels[i] for i in val_indices]
        train_pos = [grouped_position[i] for i in train_indices]
        val_pos = [grouped_position[i] for i in val_indices]
        train_ID = [grouped_playerID[i] for i in train_indices]
        val_ID = [grouped_playerID[i] for i in val_indices]

        yield train_data,  val_data, train_labels, val_labels, train_pos, val_pos, train_ID, val_ID

def match_test_train_split(org_match,test_matches):
    train_data = []
    train_labels = []
    train_pos = []
    train_ID = []
    train_names = []
    test_data = []
    test_labels = []
    test_pos = []
    test_ID = []
    test_names = []

    
    for k, (match_name, contents) in enumerate(org_match.items()):
        data = contents['data']
        labels = contents['labels']
        positions = contents['pos']
        player_ids = contents['player_id']

        if match_name in test_matches:
            test_names.append(match_name)
            for i in range(len(data)):
                test_data.append(data[i])
                test_labels.append(labels[i])
                test_pos.append(positions[i])
                test_ID.append(player_ids[i])
                #test_matches.append(match_name)
                
        else:
            train_names.append(match_name)
            for i in range(len(data)):
                train_data.append(data[i])
                train_labels.append(labels[i])
                train_pos.append(positions[i])
                train_ID.append(player_ids[i])
                #test_matches.append(match_name)
    return train_data, test_data, train_labels, test_labels, train_pos, test_pos, train_ID, test_ID
    # Process data, labels, positions, player_ids for each match

def prop_match_test_train_split(org_match,ratio= 0.20,r_state=12):
    train_data = []
    train_labels = []
    train_pos = []
    train_ID = []
    test_data = []
    test_labels = []
    test_pos = []
    test_ID = []

    
    for k, (match_name, contents) in enumerate(org_match.items()):
        data = contents['data']
        labels = contents['labels']
        positions = contents['pos']
        player_ids = contents['player_id']
        rally_indices = list(range(len(data)))
        train_indices, test_indices = train_test_split(rally_indices, test_size=ratio, random_state=r_state)
        subset_data = lambda data_type, indices: [data_type[i] for i in indices]

        train_data.extend(subset_data(data, train_indices))
        train_labels.extend(subset_data(labels, train_indices))
        train_pos.extend(subset_data(positions, train_indices))
        train_ID.extend(subset_data(player_ids, train_indices))

        test_data.extend(subset_data(data, test_indices))
        test_labels.extend(subset_data(labels, test_indices))
        test_pos.extend(subset_data(positions, test_indices))
        test_ID.extend(subset_data(player_ids, test_indices))
    return train_data, test_data, train_labels, test_labels, train_pos, test_pos, train_ID, test_ID

class player_encoding_table_db:
    def __init__(self,):
        strings = np.array(['T1P1','T2P1']) ## T1 is refering to Ginting while T2 is refering to Momota.
        print(f'Players in the dataset {strings}')
        self.player_encoding_table = {string: i for i, string in enumerate(strings)} ## plus 1 to make room for padding # no padding at the moment
        print("Encoding table:",self.player_encoding_table)
    def encode_string(self, s):
        return self.player_encoding_table.get(s, None)  # Returns None if the string is not found


def get_player_id_db(l,pet):
    if l[6] == 'T2P1':
        if l[8] == 'bottom':
            return [pet.encode_string('T2P1'),1],[pet.encode_string('T1P1'),0]
        else:
            return [pet.encode_string('T1P1'),0],[pet.encode_string('T2P1'),1]
    elif l[6] == 'T1P1':
        if l[7] == 'bottom':
            return [pet.encode_string('T1P1'),1],[pet.encode_string('T2P1'),0]
        else:
            return [pet.encode_string('T2P1'),0],[pet.encode_string('T1P1'),1]
    else:
        print('something when wrong with player encoding')
        return [None],[None]
    
def group_and_encode_badmindb_data_with_player(input_data, labels,position=None,padding_shift=False,convert=True,smoothing=True,player_splitting=False):#(input_data, labels,position=None,padding_shift=True):
    """
    Group input data and encoded labels based on rally.

    Parameters:
    input_data (np.array): Input pose sequences of shape (Batch, num_People, Timesteps, Joints, dim).
    labels (np.array): Label information of shape (Batch, labelinfo).

    Returns:
    tuple: Two lists, one for grouped input data and another for encoded shot type labels.
    """

    shot_type_encoding = {
        'Block':0, 'Block-Bh':0, 'Clear':1, 'Clear-Bh':1, 'Drive':2, 'Drive-Bh':2, 'Dropshot':3,
        'Dropshot-Bh':3, 'Flick-Serve':8, 'Net-Kill':4, 'Net-Kill-Bh':4, 'Net-Lift':5,
        'Net-Lift-Bh':5, 'Net-Shot':6, 'Net-Shot-Bh':6, 'Serve':8, 'Smash':7, 'Smash-Bh':7
    }

    pet = player_encoding_table_db()
    # Initialize a dictionary for grouping data and a list for encoded labels
    grouped_data = defaultdict(list)
    encoded_labels = defaultdict(list)
    grouped_pos = defaultdict(list)
    grouped_player_id = defaultdict(list)

    for label, pose_sequence,pos_seq in zip(labels, input_data, position):
        # Extract relevant information
        rally_number = label[1]
        match_name = label[3]
        shot_type = label[0]
        if shot_type == 'FAULT': ### Fault always last and should not be considered for this. 
            continue

        # Use a tuple of match name, set name, and rally number as the key
        key = (match_name, rally_number)

        # Group pose sequence
        if convert:
            n,t,_,_ = pose_sequence.shape
            pose_sequence = rearrange(convert_AlphaOpenposeCoco_to_standard16Joint(rearrange(pose_sequence,'n t j d -> (n t) j d')),'(n t) j d -> n t j d',n=n,t=t)
        if smoothing:
            pose_sequence = keypoint_smoothing(pose_sequence) ## smoothing after convertion if both flags applied. 

        grouped_data[key].append(rearrange(pose_sequence,'n t j d -> n t (j d)'))#.reshape(2,61,17*3))
        grouped_pos[key].append(pos_seq)
        temp = get_player_id_db(label,pet)
        grouped_player_id[key].append(np.array(temp))
        #grouped_player_id_far[key].append(temp[1])


        # Encode and group labels
        encoded_shot_type = shot_type_encoding.get(shot_type, -1)  # -1 for unknown shot types
        if padding_shift:
            encoded_labels[key].append(encoded_shot_type+1)
        else:
            encoded_labels[key].append(encoded_shot_type)
    if player_splitting:
        return grouped_data, encoded_labels, grouped_pos, grouped_player_id
    # Convert grouped data and labels into lists
    grouped_data_list = [group for group in grouped_data.values()]
    encoded_labels_list = [labels for labels in encoded_labels.values()]
    grouped_pos_list = [group for group in grouped_pos.values()]
    grouped_player_id_list = [group for group in grouped_player_id.values()]

    return grouped_data_list, encoded_labels_list, grouped_pos_list,grouped_player_id_list

def group_and_encode_shuttlenet_data_with_player(input_data, labels,position=None,match_info=None,padding_shift=False,convert=True,smoothing=True,player_splitting=False):
    """
    Group input data and encoded labels based on rally.

    Parameters:_
    input_data (np.array): Input pose sequences of shape (Batch, num_People, Timesteps, Joints, dim).
    labels (np.array): Label information of shape (Batch, labelinfo).

    Returns:
    tuple: Two lists, one for grouped input data and another for encoded shot type labels.
    """

    shot_type_encoding = {
        '放小球': 0, ## net shot 6716
        '勾球': 0, ## cross court net shot 
        '擋小球': 1, ## return net short (defensive)3836
        '防守回挑': 1, ## return/reaction lob (defensive)
        '防守回抽': 1, ## return/reaction drive (defensive)
        '殺球': 2, ## smash3749
        '點扣': 2, ## wrist smash (smash)
        '挑球': 3, ## lob 4614
        '長球': 4, ## clear 2440
        '平球': 5, ## drive 1091
        '小平球': 5, ## drive front court
        '後場抽平球': 5, ## drive to back court
        '切球': 6, ## drop 2929
        '過度切球': 6, # is this 過渡切球' and means passive drop ## I think this one was never observed...
        '推球': 7, ## push 3021
        '撲球': 7, ## rush
        '發短球': 8, ## short serve 2060
        '發長球': 8, ## long serve
        '未知球種':9, ## unknwon shot 1095
    }
   
    pet = player_encoding_table(match_info)
    # Initialize a dictionary for grouping data and a list for encoded labels
    grouped_data = defaultdict(list)
    encoded_labels = defaultdict(list)
    grouped_pos = defaultdict(list)
    grouped_player_id = defaultdict(list)


 

    for label, pose_sequence,pos_seq in zip(labels, input_data, position):
        # Extract relevant information
        rally_number = label[1]
        match_name = label[3]
        set_name = label[4]
        shot_type = label[0]

        # Use a tuple of match name, set name, and rally number as the key
        key = (match_name, set_name, rally_number)

        # Group pose sequence
        if convert:
            n,t,_,_ = pose_sequence.shape
            pose_sequence = rearrange(convert_AlphaOpenposeCoco_to_standard16Joint(rearrange(pose_sequence,'n t j d -> (n t) j d')),'(n t) j d -> n t j d',n=n,t=t)
        if smoothing:
            pose_sequence = keypoint_smoothing(pose_sequence) ## smoothing after convertion if both flags applied. 

        grouped_data[key].append(rearrange(pose_sequence,'n t j d -> n t (j d)'))#.reshape(2,61,17*3))
        grouped_pos[key].append(pos_seq)
        temp = get_player_id(label,match_info,pet)
        grouped_player_id[key].append(np.array(temp))
        #grouped_player_id_far[key].append(temp[1])


        # Encode and group labels
        encoded_shot_type = shot_type_encoding.get(shot_type, -1)  # -1 for unknown shot types
        if padding_shift:
            encoded_labels[key].append(encoded_shot_type+1)
        else:
            encoded_labels[key].append(encoded_shot_type)
    if player_splitting:
        return grouped_data, encoded_labels, grouped_pos, grouped_player_id
    # Convert grouped data and labels into lists
    grouped_data_list = [group for group in grouped_data.values()]
    encoded_labels_list = [labels for labels in encoded_labels.values()]
    grouped_pos_list = [group for group in grouped_pos.values()]
    grouped_player_id_list = [group for group in grouped_player_id.values()]

    return grouped_data_list, encoded_labels_list, grouped_pos_list,grouped_player_id_list

LMM_detailed_descriptor = {
    0:'Net shot, a precise shot with short preparation from near the net that enables theshuttlecock to tumble across just slightly above the net.',
    1:'Defensive strokes/ block, a shot that is used to return or intercept aggressive shots, often in the form of a block drive, lob or net shot. Perfomed under pressure and thus the stroke has a short preparation.',
    2:'Smash, hit far over the head, very offensive stroke often with a long preperation attempting to win the point by moving the shuttlecock in a fast downward trajectory',
    3:'Lob/lift, hit from under the net in the front court lifting the shuttlecock to the backcourt of the opponent, often a defensive/neutral shot, the stroke has a medium/long preparation.',
    4:'Clear, a floating stroke hit over the head producing a high soft trajectory aimed towards the backcourt of the opponents side, often seen as a transport shot.',
    5:'Drive, produces flat trajectory hit from the front/mid court with short perperation and hit towards middle/backcourt, high tempo stroke mostly neutral stroke attempting to gain momentum.',
    6:'Drop, a floating smooth trajectory attemping to place the shuttle very close to the net.', # Can be hit for back court or front court as a reaction to another drop or a netshot.',
    7:'Push/Rush, either a the push is a stroke hit semi softly with short perperation used to send the shuttlecock low and deep into the back of the court. Neutral shot to keep your opponent from attacking and to gain control of the court. Can also be a hard shot that strikes the shuttlecock that is too high at the net in a downward trajectory.',
    8:'Serve, starts the rally and puts the shuttlecock into play and is served toward the front service line attempting to keep the shuttle under net hight making it difficult for the opponent to attack.',
    9:'Unknown stroke, a stroke that was not recorded properly by the cameras not able to be annotated, However often occurs a the start of the rally thus some often some undocumented serve.',
}

LMM_simple_descriptor = {
    0:'Net shot',
    1:'Defensive strokes / Block, reaction',
    2:'Smash, offensiv, point finnish',
    3:'Lob/lift, high trajectory, from front',
    4:'Clear, high trajectory, from back',
    5:'Drive, flat trajectory, midcourt',
    6:'Drop, short',
    7:'Push/Rush, short prep, neutral',
    8:'Serve, Start point',
    9:'Unknown stroke',
}

from transformers import BertModel, BertTokenizer

class PoseData_Forecast(Dataset): ### for the 
    def __init__(self,dataset,labels,pos=None,shut=None,playerID=None,normalize=True,len_max = 50,
                 transform=None,factorized=False,multi_con=True,tjek_enc=False,num_joints=16,bert_model_name = 'bert-base-uncased'): ## org 75
        super().__init__()
        self.tjek_enc = tjek_enc
        self.pre = False
        self.factorized = factorized
        self.transform = transform
        self.pairs_b25 = [
            #(0,15),(0,16),(15,17),(16,18),(0,1),
            (1,2),(2,3),(3,4),
            (1,5),(5,6),(6,7),
            (1,8),(2,9),(5,12),
            (8,9),(9,10),(10,11),
            #(11,22),(11,24),(22,23),
            (8,12),(12,13),(13,14),
            #(14,21),(14,19),(19,20)
            ]

        ## head , arms, torso, legs
        self.pairs_coco = [
             (0,1),(0,2),(1,3),(2,4),
             (5,7),(7,9),(6,8),(8,10),
             (5,6),(5,11),(6,12),(11,12),
             (11,13),(13,15),(12,14),(14,16)
             ]

        ## Load from numpy
        self.num_joints = num_joints #16 # 17 
        self.clip_len = len_max
        self.n_max  = 2
        temporal = []
        persons = []
        poses = []
        pos_pad = []
        #shuttle_pad = []
        print(len(dataset))
        for m,i in enumerate(dataset):
            temporal_seq = []
            temp = np.zeros((len(i),self.n_max,self.clip_len,int(self.num_joints*3))) ## 17
            temp_pos =  np.zeros((len(i),self.n_max,self.clip_len,2))

            for j,n in enumerate(i):
                if len(n[0])<=self.clip_len:
                    temp[j,:len(n),:len(n[0])] = n[:self.n_max]
                    temp_pos[j,:len(n),:len(n[0])] = rearrange(pos[m][j],'t p x -> p t x',p=self.n_max)

                    temporal_seq.append(len(n[0]))


                elif len(n[0])>self.clip_len:
                    frames_len = len(n[0])

                    snip_loc = frames_len//self.clip_len#np.random.randint(0,frames_len-self.clip_len)
                    snip_stop = frames_len-frames_len%self.clip_len
                    temp[j,:len(n)] = n[:self.n_max, 0:snip_stop:snip_loc]
                    temp_pos[j,:len(n)] = rearrange(pos[m][j],'t p x -> p t x',p=self.n_max)[:self.n_max, 0:snip_stop:snip_loc]

                    temporal_seq.append(len_max)

            ## Position scaling
            temp_pos[:,:,:,0] = (temp_pos[:,:,:,0]-0)/300 ## court scaling factors change for other sports etc
            temp_pos[:,:,:,1] = (temp_pos[:,:,:,1]-0)/700 ## court scaling factors change for other sports etc
            
            poses.append(rearrange(temp,'s n t p -> s n t p'))
            pos_pad.append(np.array(temp_pos))
            #shuttle_pad.append(np.array(temp_shut))
            temporal.append(torch.from_numpy(np.array(temporal_seq)).type(torch.LongTensor))
        self.temporal = temporal
        if self.pre:    
            self.temporal = repeat(torch.tensor(temporal).type(torch.LongTensor), 'b -> (b k)',k=2)
            self.label = repeat(torch.LongTensor(labels),'b -> (b k)',k=2)

        self.data = poses
        B = len(poses)
        #print(self.data[0])
        S,N,T,D = self.data[0].shape
        if multi_con:
            self. data = [rearrange(torch.cat((rearrange(torch.from_numpy(normalization_prob(temp)).type(torch.FloatTensor),'s n t (j d) -> s n t j d',j=self.num_joints),
                            torch.from_numpy(pos_pad[i]).type(torch.FloatTensor).unsqueeze(3)),dim=3),'s n t j d -> s n t (j d)') for i,temp in enumerate(poses)]
        else:
            self. data = [torch.from_numpy(normalization_prob(temp)).type(torch.FloatTensor) for temp in poses]
        print(self.data[0].shape)
        self.labels = [torch.from_numpy(np.array(label)).type(torch.LongTensor) for label in labels]
        self.playerID = [torch.from_numpy(np.array(ID)).type(torch.LongTensor) for ID in playerID]
        print(self.playerID[0].shape)
        
        #self.text_description = 
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.text_description = []
        for i,strokes in enumerate(labels):
            self.text_description.append(tokenizer([LMM_simple_descriptor.get(stroke,'Invalid stroke') for stroke in strokes], padding=True, truncation=True, return_tensors="pt"))  
            #print(self.text_description[-1])
        print(len(self.text_description))
    def remove_conf(slef,data):
        placeholder = []

    def __len__(self):

        return len(self.data)
    def __getitem__(self,index):
        x_key = self.data[index]
        y_key = self.labels[index]
        
        if self.tjek_enc: 
            size = x_key.shape
            noise = torch.randn(size)
            x_key = noise
        if self.transform:
            # Apply data augmentation
            x_key = self.transform(x_key)
            #x_bones = self.transform_b(x_bones)
            #x_sp = self.transform_sp(x_sp)
        if self.factorized:
            return rearrange(x_key,'s p t d -> s p t d').type(torch.FloatTensor),y_key,self.temporal[index],self.playerID[index],self.text_description[index]#,self.sq_len[index]
        else:
            return rearrange(x_key,'s p t d -> s t (p d)').type(torch.FloatTensor),y_key,self.temporal[index],self.playerID[index],self.text_description[index]#,self.sq_len[index]