import os
from typing import Type
import torch
import torch.nn.functional as F
import pandas as pd
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from tqdm import tqdm

def attention_loss(attn_matrix):

    batch_size = attn_matrix.size(0)
    loss = 0.0

    for i in range(batch_size):
        # 对每个样本的注意力矩阵计算熵
        attn = attn_matrix[i]
        entropy = -torch.sum(attn * torch.log(attn + 1e-10), dim=-1).mean()
        loss += entropy

    return loss / batch_size

def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
    
def clip_loss(model, x, y, temperature=0.07, device='cuda'):
    
    # sim = model.stein_kernel.rbf_kernel(x, y) / temperature

    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    ecg_mu, ecg_logvar = torch.chunk(x, 2, dim=-1)
    text_mu, text_logvar = torch.chunk(y, 2, dim=-1)

    kl_loss = compute_kl_loss(ecg_mu, ecg_logvar, text_mu, text_logvar)

    # sim = torch.einsum('i d, j d -> i j', x, y) * 1 / temperature
    sim = model.stein_kernel.rbf_kernel(x, y) / temperature
    labels = torch.arange(x.shape[0]).to(device)

    loss_t = F.cross_entropy(sim, labels) 
    loss_i = F.cross_entropy(sim.T, labels) 

    i2t_acc1, i2t_acc5 = precision_at_k(
        sim, labels, top_k=(1, 5))
    t2i_acc1, t2i_acc5 = precision_at_k(
        sim.T, labels, top_k=(1, 5))
    acc1 = (i2t_acc1 + t2i_acc1) / 2.
    acc5 = (i2t_acc5 + t2i_acc5) / 2.

    return (loss_t + loss_i) + kl_loss, acc1, acc5


def compute_kl_loss(mu1, logvar1, mu2, logvar2, mask=None):
    kl1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), dim=-1)  # [B, L]
    kl2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp(), dim=-1)  # [B, L]
        
    return (kl1.mean() + kl2.mean()) 
