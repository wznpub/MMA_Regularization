import torch
import torch.nn.functional as F
import mxnet as mx


def get_mma_loss(weight):
    '''
    MMA regularization in PyTorch
    :param weight: parameter of a layer in model, out_features *　in_features
    :return: mma loss
    '''

    # for convolutional layers, flatten
    if weight.dim() > 2:
        weight = weight.view(weight.size(0), -1)

    # computing cosine similarity: dot product of normalized weight vectors
    weight_ = F.normalize(weight, p=2, dim=1)
    cosine = torch.matmul(weight_, weight_.t())

    # make sure that the diagnonal elements cannot be selected
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))

    # maxmize the minimum angle
    loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()

    return loss


def get_angular_loss_mxsymbol(weight):
    '''
    MMA regularization in Symbol of MXNet
    :param weight: parameter of a layer in model, out_features *　in_features
    :return: mma loss
    '''

    # for convolutional layers, flatten
    if 'conv' in weight.name:
        num_filter = int(weight.attr('num_filter'))
        weight = weight.reshape((num_filter, -1))
    else:
        num_filter = int(weight.attr('num_hidden'))

    # computing cosine similarity: dot product of normalized weight vectors, and make sure that the diagnonal elements cannot be selected
    weight_ = mx.symbol.L2Normalization(weight, mode='instance')
    cosine = mx.symbol.linalg.syrk(weight_, alpha=1., transpose=False) - 2. * mx.symbol.eye(num_filter)

    # maxmize the minimum angle
    theta = mx.symbol.arccos(mx.symbol.max(cosine, axis=1))
    loss = -mx.symbol.mean(theta)

    return loss
