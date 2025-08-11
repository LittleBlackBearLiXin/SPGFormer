import torch
from sklearn import metrics
import numpy as np
import spectral as spy
import matplotlib.pyplot as plt

def draw_classification_map(label, path, scale=4.0, dpi=400):
    fig, ax = plt.subplots()
    v = spy.imshow(classes=label.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    plt.savefig(path + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    plt.close()


def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
    real_labels = reallabel_onehot
    we = -torch.mul(real_labels, torch.log(predict))
    we = torch.mul(we, reallabel_mask)
    pool_cross_entropy = torch.sum(we)
    return pool_cross_entropy





@torch.no_grad()
def evaluate_performance(output, gt, onehot, class_count,
                         require_detail=True, printFlag=True, save_path=None, extra_info=""):
    #print(torch.sum(gt>0))

    valid = gt.view(-1) > 0

    pred_labels = torch.argmax(output, 1)
    #print(torch.max(pred_labels))
    #print(torch.min(pred_labels))
    true_labels = torch.argmax(onehot, 1)
    #print(torch.max(true_labels))
    #print(torch.min(true_labels))
    correct = (pred_labels == true_labels) & valid
    OA = correct.sum().float() / valid.sum().float()

    if not require_detail:
        return OA

    pred = pred_labels.cpu().numpy()[valid.cpu().numpy()]
    true = gt.view(-1).cpu().numpy()[valid.cpu().numpy()].astype(int)


    pred += 1


    kappa = metrics.cohen_kappa_score(pred, true)


    report = metrics.classification_report(
        true, pred, labels=np.arange(1, class_count + 1),
        output_dict=True, zero_division=0
    )
    acc_per_class = [report[str(i)]['recall'] for i in range(1, class_count + 1)]
    AA = np.mean(acc_per_class)


    if printFlag:
        print(f"OA={OA:.4f}, AA={AA:.4f}, Kappa={kappa:.4f}")
        print("Per-class accuracy:", acc_per_class)


    if save_path:
        with open(save_path, 'a+') as f:
            f.write("\n======================\n")
            f.write(extra_info + "\n")
            f.write(f"OA={OA:.4f}\nAA={AA:.4f}\nKappa={kappa:.4f}\n")
            f.write("Per-class acc=" + str(acc_per_class) + "\n")

    return OA, AA, kappa, acc_per_class



