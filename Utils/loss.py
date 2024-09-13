import torch
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LogMSELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LogMSELoss, self).__init__()

    def forward(self, inputs, targets, max_num, min_num):
        targets = (torch.log(targets+1) - min_num) / (max_num - min_num)
        inputs = inputs.reshape(len(targets))
        loss = (targets - inputs)**2
        return torch.mean(loss)

class DataUncertaintyLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DataUncertaintyLoss, self).__init__()

    def forward(self, pred, va, targets, max_num, min_num):
        targets = (torch.log(targets+1) - min_num) / (max_num - min_num)
        pred = pred.reshape(len(targets))
        loss = torch.log(va)/2 + torch.square(pred-targets)/(2*va) + math.log(2*math.pi)/2
        return torch.mean(loss)

class PairRankingLoss(torch.nn.Module):
    def __init__(self, margin=3.5):
        super(PairRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, cost, targets, max_num, min_num):
        n = len(targets)

        targets = (torch.log(targets + 1) - min_num) / (max_num - min_num)
        cost = cost.reshape(n)

        targets_r = targets.repeat(n, 1)
        targets_m = targets_r.transpose(0, 1) - targets_r

        cost_r = cost.repeat(n, 1)
        cost_m = cost_r.transpose(0, 1) - cost_r

        targets_t = torch.sign(targets_m)
        cost_t = torch.sign(cost_m)

        incorrect_mask = targets_t != cost_t

        loss = torch.mean(incorrect_mask * torch.exp(self.margin + torch.abs(cost_m)))

        return loss

class ExplanationLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ExplanationLoss, self).__init__()
        self.explanation_loss = torch.nn.MSELoss()

    def forward(self, expl, local_labels):
        expl = expl.reshape(-1)
        loss = self.explanation_loss(expl, local_labels)
        return loss