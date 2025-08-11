
from model_setup import ours_model_inputs, PSPT_inputs, mmpn_inputs,Combine_model_inputs




def prepare_model(MODEL, FLAG, data, gt,train_gt, val_gt, test_gt,
                  train_onehot, val_onehot, test_onehot, class_count, device):
    learning_rate = 5e-4
    WEIGHT_DECAY = 0
    max_epoch = 500
    #NN = "SGCNCNN"#CNNMPNN,SGCNCNN,SGCNPSPT,CNNPSPT

    net_input, train_gt_tensor, val_gt_tensor, test_gt_tensor = None, None, None, None
    train_onehot_tensor, val_onehot_tensor, test_onehot_tensor = None, None, None
    train_mask_tensor, val_mask_tensor, test_mask_tensor = None, None, None
    net = None
    superpixel_scale = None



    if MODEL == 'SPGformer':
        if FLAG == 1:
            superpixel_scale = 200
        else:
            superpixel_scale = 300
        net_input, \
        train_gt_tensor, val_gt_tensor, test_gt_tensor, \
        train_onehot_tensor, val_onehot_tensor, test_onehot_tensor, \
        train_mask_tensor, val_mask_tensor, test_mask_tensor, \
        net = ours_model_inputs(
            data, gt,train_gt, val_gt, test_gt,
            train_onehot, val_onehot, test_onehot,
            class_count, superpixel_scale, device,FLAG)

    elif MODEL == 'PSPT':
        if FLAG == 3:
            WEIGHT_DECAY=1e-4
        net_input, \
        train_gt_tensor, val_gt_tensor, test_gt_tensor, \
        train_onehot_tensor, val_onehot_tensor, test_onehot_tensor, \
        train_mask_tensor, val_mask_tensor, test_mask_tensor, \
        net = PSPT_inputs(
            data, gt, train_gt, val_gt, test_gt,
            train_onehot, val_onehot, test_onehot,
            class_count, superpixel_scale, device, FLAG)

    elif MODEL == 'MMPN':
        if FLAG==1:
            WEIGHT_DECAY = 1e-4
            superpixel_scale = 100
        else:
            WEIGHT_DECAY = 0
            superpixel_scale = 300
        net_input, \
        train_gt_tensor, val_gt_tensor, test_gt_tensor, \
        train_onehot_tensor, val_onehot_tensor, test_onehot_tensor, \
        train_mask_tensor, val_mask_tensor, test_mask_tensor, \
        net = mmpn_inputs(
            data, gt, train_gt, val_gt, test_gt,
            train_onehot, val_onehot, test_onehot,
            class_count, superpixel_scale, device, FLAG)

    elif MODEL == 'Combine':
        if FLAG==1:
            WEIGHT_DECAY = 1e-4
            superpixel_scale = 100
        else:
            WEIGHT_DECAY = 0
            superpixel_scale = 300
        net_input, \
        train_gt_tensor, val_gt_tensor, test_gt_tensor, \
        train_onehot_tensor, val_onehot_tensor, test_onehot_tensor, \
        train_mask_tensor, val_mask_tensor, test_mask_tensor, \
        net = Combine_model_inputs(
            data, gt, train_gt, val_gt, test_gt,
            train_onehot, val_onehot, test_onehot,
            class_count, superpixel_scale, device, FLAG,NN)
    else:
        None



    return net_input, \
           train_gt_tensor, val_gt_tensor, test_gt_tensor, \
           train_onehot_tensor, val_onehot_tensor, test_onehot_tensor, \
           train_mask_tensor, val_mask_tensor, test_mask_tensor, \
           net,\
           learning_rate, WEIGHT_DECAY, max_epoch
