import scipy.io as sio
from sklearn import preprocessing
import os
import numpy as np
import random

import torch
import matplotlib.pyplot as plt

color_map_dict = {
    'indian_': np.array([[255, 255, 255],
                         [0, 168, 132], [76, 0, 115], [0, 0, 0], [190, 255, 232], [255, 0, 0],
                         [115, 0, 0], [205, 205, 102], [137, 90, 68], [215, 158, 158], [255, 115, 223],
                         [0, 0, 255], [156, 156, 156], [115, 223, 255], [0, 255, 0], [255, 255, 0],
                         [255, 170, 0]], dtype=np.uint8),

    'paviaU_': np.array([[255, 255, 255],
                         [0, 0, 255], [76, 230, 0], [255, 190, 232], [255, 0, 0], [156, 156, 156],
                         [255, 255, 115], [0, 255, 197], [132, 0, 168], [0, 0, 0]], dtype=np.uint8),

    'salinas_': np.array([[255, 255, 255],
                          [0, 168, 132], [76, 0, 115], [0, 0, 0], [190, 255, 232], [255, 0, 0],
                          [115, 0, 0], [205, 205, 102], [137, 90, 68], [215, 158, 158], [255, 115, 223],
                          [0, 0, 255], [156, 156, 156], [115, 223, 255], [0, 255, 0], [255, 255, 0],
                          [255, 170, 0]], dtype=np.uint8),
}




def pltgraph(gt,output,name,dataset_name,truegarph=False):
    height, width= gt.shape
    if truegarph:
        classification_map = gt
    else:
        predy = torch.argmax(output, 1).reshape([height, width]).cpu() + 1
        classification_map = (torch.where(torch.tensor(gt) > 0, 1, 0)) * predy
    #print("Max value in classification_map:", classification_map.max())
    #print("Min value in classification_map:", classification_map.min())
    palette = color_map_dict.get(dataset_name)
    map = palette[classification_map]
    plt.figure()
    plt.imshow(map, cmap='jet')
    plt.xticks([])
    plt.yticks([])

    output_dir = dataset_name
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, name + '.png'),
        bbox_inches='tight', dpi=300)

    print('sava to:',dataset_name,name)









def load_dataset(flag):
    parent_dir = os.path.dirname(os.path.abspath(__file__))

    if flag == 1:
        data_path = os.path.join(parent_dir, '../HyperImage_data/indian/Indian_pines_corrected.mat')
        gt_path = os.path.join(parent_dir, '../HyperImage_data/indian/Indian_pines_gt.mat')
        data = sio.loadmat(data_path)['indian_pines_corrected']
        gt = sio.loadmat(gt_path)['indian_pines_gt']
        class_count = 16
        dataset_name = "indian_"
        print('Useing data is:',dataset_name)
    elif flag == 2:
        data_path = os.path.join(parent_dir, '../HyperImage_data/paviaU/PaviaU.mat')
        gt_path = os.path.join(parent_dir, '../HyperImage_data/paviaU/PaviaU_gt.mat')
        data = sio.loadmat(data_path)['paviaU']
        gt = sio.loadmat(gt_path)['paviaU_gt']
        class_count = 9
        dataset_name = "paviaU_"
        print('Useing data is:', dataset_name)
    elif flag == 3:
        data_path = os.path.join(parent_dir, '../HyperImage_data/Salinas/Salinas_corrected.mat')
        gt_path = os.path.join(parent_dir, '../HyperImage_data/Salinas/Salinas_gt.mat')
        data = sio.loadmat(data_path)['salinas_corrected']
        gt = sio.loadmat(gt_path)['salinas_gt']
        class_count = 16
        dataset_name = "salinas_"
        print('Useing data is:', dataset_name)
    else:
        raise ValueError("Invalid FLAG value")
    data = standardize_data(data)

    return data, gt, class_count, dataset_name


def standardize_data(data):
    height, width, bands = data.shape
    reshaped = data.reshape(-1, bands)
    scaler = preprocessing.StandardScaler()
    normalized = scaler.fit_transform(reshaped).reshape(height, width, bands)
    return normalized


def normalize_data(data):

    min_val = np.min(data)
    max_val = np.max(data)


    normalized = (data - min_val) / (max_val - min_val)

    return normalized




def ours_split(gt, seed, class_count):
    train_samples_per_class = 5
    val_samples_per_class = 5
    height, width = gt.shape
    total = height * width
    gt_reshape = gt.reshape(-1)
    random.seed(seed)


    train_idx = []
    for c in range(1, class_count + 1):
        idx = np.where(gt_reshape == c)[0]
        if len(idx) == 0:
            continue
        num = min(train_samples_per_class, len(idx) // 2)
        sampled = random.sample(list(idx), num)
        train_idx.extend(sampled)

    train_idx = set(train_idx)
    all_idx = set(np.where(gt_reshape > 0)[0])
    test_idx = all_idx - train_idx


    val_idx = set()
    for c in range(1, class_count + 1):
        class_test_idx = [i for i in test_idx if gt_reshape[i] == c]
        if len(class_test_idx) >= val_samples_per_class:
            sampled_val = random.sample(class_test_idx, val_samples_per_class)
            val_idx.update(sampled_val)

    test_idx -= val_idx


    def build_mask(index_set):
        mask = np.zeros(total)
        for i in index_set:
            mask[i] = gt_reshape[i]
        return mask.reshape(height, width)

    train_gt = build_mask(train_idx)
    val_gt = build_mask(val_idx)
    test_gt = build_mask(test_idx)


    def to_one_hot(gt_img):
        onehot = np.zeros((total, class_count), dtype=np.float32)
        flat = gt_img.reshape(-1)
        for i in range(total):
            if flat[i] > 0:
                onehot[i, int(flat[i]) - 1] = 1
        return onehot

    return train_gt, val_gt, test_gt, to_one_hot(train_gt), to_one_hot(val_gt), to_one_hot(test_gt)



def split_data(gt, seed, class_count):

    train_gt, val_gt, test_gt, train_onehot, val_onehot, test_onehot = ours_split(gt, seed, class_count)

    return train_gt, val_gt, test_gt, train_onehot, val_onehot, test_onehot

