
from model_setup import ours_model_inputs, PSPT_inputs, mmpn_inputs,Combine_model_inputs




def prepare_model(MODEL, FLAG, data, gt,train_gt, val_gt, test_gt,
                  train_onehot, val_onehot, test_onehot, class_count, device):
    learning_rate = 5e-4
    WEIGHT_DECAY = 0
    max_epoch = 500

    net_input, train_gt_tensor, val_gt_tensor, test_gt_tensor = None, None, None, None
    train_onehot_tensor, val_onehot_tensor, test_onehot_tensor = None, None, None
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
        net = ours_model_inputs(
            data, gt,train_gt, val_gt, test_gt,
            train_onehot, val_onehot, test_onehot,
            class_count, superpixel_scale, device,FLAG)
    else:
        None



    return net_input, \
           train_gt_tensor, val_gt_tensor, test_gt_tensor, \
           train_onehot_tensor, val_onehot_tensor, test_onehot_tensor, \
           net,\
           learning_rate, WEIGHT_DECAY, max_epoch



