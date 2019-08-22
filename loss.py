import torch
import torch.nn as nn
import torch.nn.functional as F

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image, mask):
    # https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/loss.py
    loss = (torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]) * (1 - mask[:,:,:,:-1])).mean() + \
        (torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]) * (1 - mask[:,:,:-1,:])).mean()

    return loss

def perceptual_loss_func(out_features, gt_features, comp_features):
    total_loss = 0

    for (out, gt, comp) in zip(out_features, gt_features, comp_features):
        total_loss += torch.abs(out - gt).mean() + torch.abs(comp - gt).mean()

    return total_loss

class Loss(nn.Module):
    def __init__(self, extractor):
        super(Loss, self).__init__()

        self.extractor = extractor

    # input to network, mask, output of network, target
    def forward(self, input, mask, output, gt):
        comp = (1 - mask) * output + mask * gt

        out_features = self.extractor(output)
        gt_features = self.extractor(gt)
        comp_features = self.extractor(comp)

        hole_loss = torch.abs((1 - mask) * (output - gt)).mean()
        valid_loss = torch.abs(mask * (output - gt)).mean()
        perceptual_loss = perceptual_loss_func(out_features, gt_features, comp_features)

        style_loss = perceptual_loss_func([gram_matrix(feature) for feature in out_features],
                                    [gram_matrix(feature) for feature in gt_features],
                                    [gram_matrix(feature) for feature in comp_features])

        
        tv_loss = total_variation_loss(comp, mask)

        return valid_loss + 6 * hole_loss + 0.05 * perceptual_loss + 120 * style_loss + 0.1 * tv_loss


