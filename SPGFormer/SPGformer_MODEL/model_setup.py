import numpy as np
import torch
import time
from layers.getmask import process_mask, PSPT_process_mask
from SLIC.our_SLIC import Segmenttttt
from layers.SPGformer import SPGformer
from layers.pspt import Pspaformer
from layers.mmpn import mumpnnn





def ours_model_inputs(data,gt,
                         train_gt, val_gt, test_gt,
                         train_onehot, val_onehot, test_onehot,
                         class_count, superpixel_scale, device,FLAG):
    print('useing SPGformer')
    height, width, bands = data.shape
    m, n = height, width
    total = m * n

    def create_mask(gt_2d):
        mask = np.zeros((total, class_count), dtype=np.float32)
        flat = gt_2d.reshape(-1)
        for i in range(total):
            if flat[i] != 0:
                mask[i] = np.ones(class_count, dtype=np.float32)
        return mask

    train_mask = create_mask(train_gt)
    val_mask = create_mask(val_gt)
    test_mask = create_mask(test_gt)


    ls = Segmenttttt(data, class_count - 1)
    tic0 = time.perf_counter()
    Q, A = ls.SLIC_Process(data, scale=superpixel_scale)
    toc0 = time.perf_counter()
    LDA_SLIC_Time = toc0 - tic0
    print("LDA-SLIC costs time: ", LDA_SLIC_Time)
    Q = torch.from_numpy(Q).to(device)
    A = torch.from_numpy(A).to(device)




    def to_tensor(x):
        return torch.from_numpy(x.astype(np.float32)).to(device)

    train_gt_tensor = to_tensor(train_gt.reshape(-1))
    val_gt_tensor = to_tensor(val_gt.reshape(-1))
    test_gt_tensor = to_tensor(test_gt.reshape(-1))

    train_onehot_tensor = to_tensor(train_onehot)
    val_onehot_tensor = to_tensor(val_onehot)
    test_onehot_tensor = to_tensor(test_onehot)

    train_mask_tensor = to_tensor(train_mask)
    val_mask_tensor = to_tensor(val_mask)
    test_mask_tensor = to_tensor(test_mask)

    net_input = to_tensor(np.array(data, dtype=np.float32))
    if FLAG == 1:
        name = 'indian_'
    elif FLAG == 2:
        name = 'paviaU_'
    elif FLAG == 3:
        name = 'salinas_'

    ticous = time.perf_counter()

    rowmask,colmask = process_mask(net_input, device, FLAG=FLAG)
    tocous = time.perf_counter()
    print('ous get mask time:', (tocous - ticous) + LDA_SLIC_Time)

    print('useing former')
    net = SPGformer(height=height, width=width, changel=bands, class_count=class_count, rowmask=rowmask, colmask=colmask ,Q=Q, A=A,flag=FLAG)

    net.to(device)

    def count_parameters(net):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Model parameters:", count_parameters(net))

    return (net_input,
            train_gt_tensor, val_gt_tensor, test_gt_tensor,
            train_onehot_tensor, val_onehot_tensor, test_onehot_tensor,
            train_mask_tensor, val_mask_tensor, test_mask_tensor,
            net)





def PSPT_inputs(data,gt,
                         train_gt, val_gt, test_gt,
                         train_onehot, val_onehot, test_onehot,
                         class_count, superpixel_scale, device,FLAG):
    print('useing PSPT')
    height, width, bands = data.shape
    m, n = height, width
    total = m * n

    def create_mask(gt_2d):
        mask = np.zeros((total, class_count), dtype=np.float32)
        flat = gt_2d.reshape(-1)
        for i in range(total):
            if flat[i] != 0:
                mask[i] = np.ones(class_count, dtype=np.float32)
        return mask

    train_mask = create_mask(train_gt)
    val_mask = create_mask(val_gt)
    test_mask = create_mask(test_gt)



    def to_tensor(x):
        return torch.from_numpy(x.astype(np.float32)).to(device)

    train_gt_tensor = to_tensor(train_gt.reshape(-1))
    val_gt_tensor = to_tensor(val_gt.reshape(-1))
    test_gt_tensor = to_tensor(test_gt.reshape(-1))

    train_onehot_tensor = to_tensor(train_onehot)
    val_onehot_tensor = to_tensor(val_onehot)
    test_onehot_tensor = to_tensor(test_onehot)

    train_mask_tensor = to_tensor(train_mask)
    val_mask_tensor = to_tensor(val_mask)
    test_mask_tensor = to_tensor(test_mask)

    net_input = to_tensor(np.array(data, dtype=np.float32))
    if FLAG == 1:
       hide = 128
       layers_count = 2
    elif FLAG == 2:
        hide = 64
        layers_count = 2
    elif FLAG == 3:
        hide = 64
        layers_count = 3
    else:
        hide = 64
        layers_count = 2

    ticous = time.perf_counter()

    rowmask,colmask = PSPT_process_mask(net_input, device, FLAG=FLAG)
    tocous = time.perf_counter()
    print('ous get mask time:', (tocous - ticous) )

    print('useing former')
    net = Pspaformer(height=height, width=width, changel=bands, class_count=class_count, rowmask=rowmask, colmask=colmask,hide=hide,layers_count=layers_count)

    net.to(device)

    def count_parameters(net):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Model parameters:", count_parameters(net))

    return (net_input,
            train_gt_tensor, val_gt_tensor, test_gt_tensor,
            train_onehot_tensor, val_onehot_tensor, test_onehot_tensor,
            train_mask_tensor, val_mask_tensor, test_mask_tensor,
            net)




def mmpn_inputs(data,gt,
                         train_gt, val_gt, test_gt,
                         train_onehot, val_onehot, test_onehot,
                         class_count, superpixel_scale, device,FLAG):
    print('useing MMPN')
    height, width, bands = data.shape
    m, n = height, width
    total = m * n

    def create_mask(gt_2d):
        mask = np.zeros((total, class_count), dtype=np.float32)
        flat = gt_2d.reshape(-1)
        for i in range(total):
            if flat[i] != 0:
                mask[i] = np.ones(class_count, dtype=np.float32)
        return mask

    train_mask = create_mask(train_gt)
    val_mask = create_mask(val_gt)
    test_mask = create_mask(test_gt)


    ls = Segmenttttt(data, class_count - 1)
    tic0 = time.perf_counter()
    Q, A = ls.SLIC_Process(data, scale=superpixel_scale)
    toc0 = time.perf_counter()
    LDA_SLIC_Time = toc0 - tic0
    print("LDA-SLIC costs time: ", LDA_SLIC_Time)
    Q = torch.from_numpy(Q).to(device)
    A = torch.from_numpy(A).to(device)




    def to_tensor(x):
        return torch.from_numpy(x.astype(np.float32)).to(device)

    train_gt_tensor = to_tensor(train_gt.reshape(-1))
    val_gt_tensor = to_tensor(val_gt.reshape(-1))
    test_gt_tensor = to_tensor(test_gt.reshape(-1))

    train_onehot_tensor = to_tensor(train_onehot)
    val_onehot_tensor = to_tensor(val_onehot)
    test_onehot_tensor = to_tensor(test_onehot)

    train_mask_tensor = to_tensor(train_mask)
    val_mask_tensor = to_tensor(val_mask)
    test_mask_tensor = to_tensor(test_mask)

    net_input = to_tensor(np.array(data, dtype=np.float32))
    if FLAG == 1:
        name = 'indian_'
    elif FLAG == 2:
        name = 'paviaU_'
    elif FLAG == 3:
        name = 'salinas_'

    print('ous get mask time:', LDA_SLIC_Time)

    print('useing former')
    net = mumpnnn(height=height, width=width, changel=bands, class_count=class_count ,Q=Q, A=A,flag=FLAG)

    net.to(device)

    def count_parameters(net):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Model parameters:", count_parameters(net))

    return (net_input,
            train_gt_tensor, val_gt_tensor, test_gt_tensor,
            train_onehot_tensor, val_onehot_tensor, test_onehot_tensor,
            train_mask_tensor, val_mask_tensor, test_mask_tensor,
            net)




from layers.Combine import Combine
def Combine_model_inputs(data,gt,
                         train_gt, val_gt, test_gt,
                         train_onehot, val_onehot, test_onehot,
                         class_count, superpixel_scale, device,FLAG,MODEL):
    print('useing COMbine')
    height, width, bands = data.shape
    m, n = height, width
    total = m * n

    def create_mask(gt_2d):
        mask = np.zeros((total, class_count), dtype=np.float32)
        flat = gt_2d.reshape(-1)
        for i in range(total):
            if flat[i] != 0:
                mask[i] = np.ones(class_count, dtype=np.float32)
        return mask

    train_mask = create_mask(train_gt)
    val_mask = create_mask(val_gt)
    test_mask = create_mask(test_gt)


    ls = Segmenttttt(data, class_count - 1)
    tic0 = time.perf_counter()
    Q, A = ls.SLIC_Process(data, scale=superpixel_scale)
    toc0 = time.perf_counter()
    LDA_SLIC_Time = toc0 - tic0
    print("LDA-SLIC costs time: ", LDA_SLIC_Time)
    Q = torch.from_numpy(Q).to(device)
    A = torch.from_numpy(A).to(device)




    def to_tensor(x):
        return torch.from_numpy(x.astype(np.float32)).to(device)

    train_gt_tensor = to_tensor(train_gt.reshape(-1))
    val_gt_tensor = to_tensor(val_gt.reshape(-1))
    test_gt_tensor = to_tensor(test_gt.reshape(-1))

    train_onehot_tensor = to_tensor(train_onehot)
    val_onehot_tensor = to_tensor(val_onehot)
    test_onehot_tensor = to_tensor(test_onehot)

    train_mask_tensor = to_tensor(train_mask)
    val_mask_tensor = to_tensor(val_mask)
    test_mask_tensor = to_tensor(test_mask)

    net_input = to_tensor(np.array(data, dtype=np.float32))
    if FLAG == 1:
       hide = 128
       layers_count = 2
    elif FLAG == 2:
        hide = 128
        layers_count = 2
    elif FLAG == 3:
        hide = 128
        layers_count = 3
    else:
        hide = 64
        layers_count = 2

    ticous = time.perf_counter()
    if MODEL=="SGCNPSPT":
        rowmask, colmask = process_mask(net_input, device, FLAG=FLAG)
    else:
        rowmask,colmask = PSPT_process_mask(net_input, device, FLAG=FLAG)
    tocous = time.perf_counter()
    print('ous get mask time:', (tocous - ticous) )

    print('useing former')
    net = Combine(height=height, width=width, changel=bands, class_count=class_count, rowmask=rowmask, colmask=colmask,
                  Q=Q,A=A,hide=hide,flag=FLAG,model=MODEL)

    net.to(device)

    def count_parameters(net):
        return sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Model parameters:", count_parameters(net))

    return (net_input,
            train_gt_tensor, val_gt_tensor, test_gt_tensor,
            train_onehot_tensor, val_onehot_tensor, test_onehot_tensor,
            train_mask_tensor, val_mask_tensor, test_mask_tensor,
            net)
