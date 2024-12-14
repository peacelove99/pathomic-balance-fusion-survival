import torch
from torch import nn
import torch.nn.functional as F


def define_loss(args):
    if args.loss == "ce_surv":
        loss = CrossEntropySurvLoss(alpha=0.0)
    elif args.loss == "nll_surv":
        loss = NLLSurvLoss(alpha=0.0)
    elif args.loss == "cox_surv":
        loss = CoxSurvLoss()
    elif args.loss == "nll_surv_kl":
        print('########### ', "nll_surv_kl")
        loss = [NLLSurvLoss(alpha=0.0), KLLoss()]
    elif args.loss == "nll_surv_mse":
        print('########### ', "nll_surv_mse")
        loss = [NLLSurvLoss(alpha=0.0), nn.MSELoss()]
    elif args.loss == "nll_surv_l1":
        print('########### ', "nll_surv_l1")
        loss = [NLLSurvLoss(alpha=0.0), nn.L1Loss()]
    elif args.loss == "nll_surv_cos":
        print('########### ', "nll_surv_cos")
        loss = [NLLSurvLoss(alpha=0.0), CosineLoss()]
    elif args.loss == "nll_surv_ol":
        print('########### ', "nll_surv_ol")
        loss = [NLLSurvLoss(alpha=0.0), OrthogonalLoss(gamma=0.5)]
    else:
        raise NotImplementedError
    return loss


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)


class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)


class KLLoss(object):
    def __call__(self, y, y_hat):
        return F.kl_div(y_hat.softmax(dim=-1).log(), y.softmax(dim=-1), reduction="sum")


class CosineLoss(object):
    def __call__(self, y, y_hat):
        return 1 - F.cosine_similarity(y, y_hat, dim=1)


class OrthogonalLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, P, P_hat, G, G_hat):
        pos_pairs = (1 - torch.abs(F.cosine_similarity(P.detach(), P_hat, dim=1))) + (
            1 - torch.abs(F.cosine_similarity(G.detach(), G_hat, dim=1))
        )
        neg_pairs = (
            torch.abs(F.cosine_similarity(P, G, dim=1))
            + torch.abs(F.cosine_similarity(P.detach(), G_hat, dim=1))
            + torch.abs(F.cosine_similarity(G.detach(), P_hat, dim=1))
        )

        loss = pos_pairs + self.gamma * neg_pairs
        return loss


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7, reduction='sum'):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()

    S_padded = torch.cat([torch.ones_like(c), S], 1)

    s_prev = torch.gather(S_padded, dim=1, index=Y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=Y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=Y + 1).clamp(min=eps)

    # c = 1 means censored. Weight 0 in this case
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)

    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7, reduction='sum'):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()

    S_padded = torch.cat([torch.ones_like(c), S], 1)

    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y) + eps) +
                      torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = (-c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) -
            (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps)))

    loss = (1 - alpha) * ce_l + alpha * reg

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss
