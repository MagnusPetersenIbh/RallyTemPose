from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader, random_split
from einops.layers.torch import Rearrange
import math
import torch
from sklearn.metrics import accuracy_score
import numpy as np
import os
def euclid_dist(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)
    
def normalization_prob(inp,conf=False): # should specify for confidence level / 3D aswell
    S,N,T,D = inp.shape
    x = inp.copy().reshape(-1,inp.shape[2],inp.shape[3])
    for batch in x:
        for t in batch:
            min_x = 2000
            min_y = 2000
            for v in t[0::3]:
                if v !=0 and v<min_x:
                    min_x = v
            for v in t[1::3]:
                if v !=0 and v<min_y:
                    min_y = v
            max_x = np.amax(t[0::3])
            max_y = np.amax(t[1::3])
            dist = euclid_dist((min_x,min_y),(max_x,max_y))
            if dist > 0:
                for i in range(0,len(t),3): #(if t_i >0.0) yes
                    if t[i]>0.:
                        t[i] = (t[i]-min_x)/dist
                    if t[i+1]>0.:
                        t[i+1] = (t[i+1]-min_y)/dist
    if not conf:
        x = x.reshape(-1,T,int(D/3),3)[:,:,:,:2]
        x = x.reshape(-1,T,int(2*D/3))
    return x.reshape(S,N,T,int(2*D/3))


def adjust_lr(optimizer,lr_max,lr_min,epoch,num_epochs,warmup_epochs):
    if epoch < warmup_epochs:
        lr = lr_max * epoch / warmup_epochs
    else:
        lr = lr_min + 0.5 * (lr_max-lr_min) * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs))) 
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr



def load_best_model(model, optimizer, filename="best_model.pt",device='cpu'):
    """
    Load the best model from the checkpoint if it exists.
    """
    if os.path.isfile(filename):
        print("Loading best model...")
        checkpoint = torch.load(filename,map_location=device)
        # model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint.get('best_accuracy', -1)
    return -1


def save_best_model(model, optimizer, best_accuracy, filename="best_model.pt"):
    """
    Save the model as the best model checkpoint.
    """
    print("Saving new best model...")

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_accuracy": best_accuracy,
        }
    torch.save(checkpoint, filename)


### custom accuracy/loss section 
def custom_top_k_accuracy(output, target, k=1, ignore_index=None):
    """
    Computes top-k accuracy, excluding samples with target equal to ignore_index.
    Args:
        output (torch.Tensor): The output logits from the model (size: batch_size x num_classes).
        target (torch.Tensor): The ground truth labels (size: batch_size).
        k (int): The top 'k' predictions to consider for accuracy.
        ignore_index (int, optional): The label index to ignore in accuracy calculation.
    Returns:
        float: The top-k accuracy.
    """
    with torch.no_grad():
        # Exclude samples with the ignore_index
        if ignore_index is not None:
            mask = target != ignore_index
            output = output[mask]
            target = target[mask]

        # Top-k predictions
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        # Compute top-k accuracy

        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        if  target.size(0)!=0:
            accuracy = correct_k.mul_(100.0 / target.size(0))
        else:
            accuracy = 0
            return accuracy
        return accuracy.item()


def custom_accuracy_score(y_pred,y_true, ignore_index=None):
    if ignore_index is not None:
        # Create a mask for elements to keep (those not equal to ignore_index)
        mask = y_true != ignore_index

        # Filter both y_true and y_pred using the mask
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
    else:
        y_true_filtered, y_pred_filtered = y_true, y_pred

    # Calculate accuracy using scikit-learn's function on filtered data
    return accuracy_score(y_true_filtered, y_pred_filtered)


def main(): ## move the model name in the beginning to here and change just the two depth parameters for now but we will make a
    # Instantiate your model with the specified configuration
    pass

if __name__ == "__main__":
    main()
