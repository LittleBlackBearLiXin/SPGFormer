import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, find_boundaries
from skimage.io import imread, imsave
from skimage.transform import resize
from sklearn import preprocessing
import torch
import seaborn as sns
import os
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import random

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

fix_seed(42)




def graph(adj, name,dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(adj.numpy(), cmap='Blues', annot=False, cbar=True)

    # Add title and labels
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')

    save_path = dir
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    plt.close()
    print(f"Figure saved to {save_path}")  #

def SegmentsLabelProcess(labels):

    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels


class SLICcccc(object):
    def __init__(self, HSI, n_segments=1000, compactness=0.01, max_iter=10, sigma=0, min_size_factor=0.3,
                 max_size_factor=2):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        height, width, bands = HSI.shape  #
        data = np.reshape(HSI, [height * width, bands])
        min_max = preprocessing.StandardScaler()
        data = min_max.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands])


    def get_Q_and_S_and_Segments(self):
        img = self.data
        (h, w, d) = img.shape
        segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, max_num_iter=self.max_iter,
                        convert2lab=False, sigma=self.sigma, enforce_connectivity=True,
                        min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor, slic_zero=False)

        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
            segments = SegmentsLabelProcess(segments)
        self.segments = segments


        superpixel_count = segments.max() + 1  # =segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)

        segments_1 = np.reshape(segments, [-1])
        S = np.zeros([superpixel_count, d], dtype=np.float32)  # 超像素特征

        Q = np.zeros([w * h, superpixel_count], dtype=np.float32)  # 像素与超像素联系矩阵
        x = np.reshape(img, [-1, d])  # Flatten(x)

        x_center = np.zeros([superpixel_count], dtype=np.float32)
        y_center = np.zeros([superpixel_count], dtype=np.float32)

        for i in range(superpixel_count):
            idx = np.where(segments_1 == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            S[i] = superpixel
            Q[idx, i] = 1

            seg_idx = np.where(segments == i)
            x_center[i] = np.mean(seg_idx[0])
            y_center[i] = np.mean(seg_idx[1])

        self.S = S
        self.Q = Q



        return Q


    def get_A(self):

        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        centr_pos = np.zeros([self.superpixel_count, 2], dtype=np.float32)  # 每个超像素的质心坐标 (x, y)
        (h, w) = self.segments.shape
        #print(self.segments)
        #print(self.segments.shape)


        for i in range(h - 1):
            for j in range(w - 1):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                # if len(sub_set)>1:
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                         continue
                    A[idx1, idx2] = A[idx2, idx1] = 1

        return A


class Segmenttttt(object):
    def __init__(self, data, n_component):
        self.data = data
        self.n_component = n_component
        self.height, self.width, self.bands = data.shape


    def SLIC_Process(self, img, scale=25):
        n_segments_init = self.height * self.width / scale
        print("n_segments_init", n_segments_init)

        myslic = SLICcccc(img, n_segments=n_segments_init, sigma=1, min_size_factor=0.1,
                      max_size_factor=10)

        Q= myslic.get_Q_and_S_and_Segments()
        A = myslic.get_A()
        return Q,  A






def noramlize(A,random_wolk=False):
    A = A + torch.eye(A.shape[0], A.shape[0]).to(device)
    if random_wolk:
        return F.normalize(A,dim=1,p=1)
    else:
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        A = torch.mm(torch.mm(D_hat, A), D_hat)
        return A

def spareaA(adjacency):
    adjacency=noramlize(adjacency)
    if not adjacency.is_sparse:
        adjacency = adjacency.to_sparse()
    adjacency = adjacency.coalesce()
    indices = adjacency.indices()
    values = adjacency.values()
    size = adjacency.size()
    adjacency = torch.sparse_coo_tensor(indices, values, size, dtype=torch.float32, device=device)
    return adjacency


def Ahop(adjacency, K):
    result = adjacency.clone()
    for i in range(K - 1):
        result = result @ adjacency
    return result

def getimportA(adjacency, k=2, FLAG=0):
    imadjaceny = Ahop(adjacency, k)
    if FLAG == 1:
        imadjaceny = imadjaceny * adjacency
    else:
        imadjaceny = imadjaceny
    imadjaceny = torch.where(imadjaceny > k-1, 1, 0).float()
    return imadjaceny
