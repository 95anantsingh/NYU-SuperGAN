import math
import torch
import torchvision
import torch.nn as nn
# from data.config import TestBaseTransform, widerface_640 as cfg
from face_detection_dsfd.data import TestBaseTransform, widerface_640 as cfg
# from layers import Detect, get_prior_boxes, FEM, pa_multibox, mio_module, upsample_product
from face_detection_dsfd.layers import *
# from utils import resize_image
import torch.nn.functional as F


class DeepHeadModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DeepHeadModule, self).__init__()
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._mid_channels = min(self._input_channels, 256)

        self.conv1 = nn.Conv2d(self._input_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self._mid_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self._mid_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv4 = nn.Conv2d(self._mid_channels, self._output_channels, kernel_size=1, dilation=1, stride=1,
                               padding=0)

    def forward(self, x):
        return self.conv4(
            F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x), inplace=True)), inplace=True)), inplace=True))


class FEM(nn.Module):
    def __init__(self, channel_size):
        super(FEM, self).__init__()
        self.cs = channel_size
        self.cpm1 = nn.Conv2d(self.cs, 256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm2 = nn.Conv2d(self.cs, 256, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm3 = nn.Conv2d(256, 128, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm4 = nn.Conv2d(256, 128, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm5 = nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1)

    def forward(self, x):
        x1_1 = F.relu(self.cpm1(x), inplace=True)
        x1_2 = F.relu(self.cpm2(x), inplace=True)
        x2_1 = F.relu(self.cpm3(x1_2), inplace=True)
        x2_2 = F.relu(self.cpm4(x1_2), inplace=True)
        x3_1 = F.relu(self.cpm5(x2_2), inplace=True)
        return torch.cat((x1_1, x2_1, x3_1), 1)


def upsample_product(x, y):
    '''Upsample and add two feature maps.
       Args:
         x: (Variable) top feature map to be upsampled.
         y: (Variable) lateral feature map.
       Returns:
         (Variable) added feature map.
       Note in PyTorch, when input size is odd, the upsampled feature map
       with `F.upsample(..., scale_factor=2, mode='nearest')`
       maybe not equal to the lateral feature map size.
       e.g.
       original input size: [N,_,15,15] ->
       conv2d feature map size: [N,_,8,8] ->
       upsampled feature map size: [N,_,16,16]
       So we choose bilinear upsample which supports arbitrary output sizes.
       '''
    _, _, H, W = y.size()

    # FOR ONNX CONVERSION
    # return F.interpolate(x, scale_factor=2, mode='nearest') * y
    return F.interpolate(x, size=(int(H), int(W)), mode='bilinear', align_corners=False) * y


def pa_multibox(output_channels):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(output_channels):
        if k == 0:
            loc_output = 4
            conf_output = 2
        elif k == 1:
            loc_output = 8
            conf_output = 4
        else:
            loc_output = 12
            conf_output = 6
        loc_layers += [DeepHeadModule(512, loc_output)]
        conf_layers += [DeepHeadModule(512, (2 + conf_output))]
    return (loc_layers, conf_layers)


def mio_module(each_mmbox, len_conf, your_mind_state='peasant'):
    # chunk = torch.split(each_mmbox, 1, 1) - !!!!! failed to export on PyTorch v1.0.1 (ONNX version 1.3)
    chunk = torch.chunk(each_mmbox, int(each_mmbox.shape[1]), 1)

    # some hacks for ONNX and Inference Engine export
    if your_mind_state == 'peasant':
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
    elif your_mind_state == 'advanced':
        bmax = torch.max(each_mmbox[:, :3], 1)[0].unsqueeze(0)
    else: # supermind
        bmax = torch.nn.functional.max_pool3d(each_mmbox[:, :3], kernel_size=(3, 1, 1))

    cls = (torch.cat((bmax, chunk[3]), dim=1) if len_conf == 0 else torch.cat((chunk[3], bmax), dim=1))
    cls = torch.cat((cls, *list(chunk[4:])), dim=1)
    return cls


class Detect(nn.Module):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh, variance=(0.1, 0.2)):
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = variance

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)

        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            default = prior_data
            decoded_boxes = decode(loc_data[i], default, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0 or scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    # (cx,cy,w,h)->(x0,y0,x1,y1)
    return boxes


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1.resize_(0))
        torch.index_select(y1, 0, idx, out=yy1.resize_(0))
        torch.index_select(x2, 0, idx, out=xx2.resize_(0))
        torch.index_select(y2, 0, idx, out=yy2.resize_(0))
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def get_prior_boxes(cfg, feature_maps, image_size):

    # number of priors for feature map location (either 4 or 6)
    variance = cfg['variance'] or [0.1]
    min_sizes = cfg['min_sizes']
    max_sizes = cfg['max_sizes']
    steps = cfg['steps']
    aspect_ratios = cfg['aspect_ratios']
    clip = cfg['clip']
    for v in variance:
        if v <= 0:
            raise ValueError('Variances must be greater than 0')

    mean = []

    if len(min_sizes) == 5:
        feature_maps = feature_maps[1:]
        steps = steps[1:]
    if len(min_sizes) == 4:
        feature_maps = feature_maps[2:]
        steps = steps[2:]

    for k, f in enumerate(feature_maps):
        # for i, j in product(range(f), repeat=2):
        for i in range(f[0]):
            for j in range(f[1]):
                # f_k = image_size / steps[k]
                f_k_i = image_size[0] / steps[k]
                f_k_j = image_size[1] / steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k_j
                cy = (i + 0.5) / f_k_i
                # aspect_ratio: 1
                # rel size: min_size
                s_k_i = min_sizes[k] / image_size[1]
                s_k_j = min_sizes[k] / image_size[0]
                # swordli@tencent
                if len(aspect_ratios[0]) == 0:
                    mean += [cx, cy, s_k_i, s_k_j]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                # s_k_prime = sqrt(s_k * (max_sizes[k]/image_size))
                if len(max_sizes) == len(min_sizes):
                    s_k_prime_i = math.sqrt(s_k_i * (max_sizes[k] / image_size[1]))
                    s_k_prime_j = math.sqrt(s_k_j * (max_sizes[k] / image_size[0]))
                    mean += [cx, cy, s_k_prime_i, s_k_prime_j]
                # rest of aspect ratios
                for ar in aspect_ratios[k]:
                    if len(max_sizes) == len(min_sizes):
                        mean += [cx, cy, s_k_prime_i / math.sqrt(ar), s_k_prime_j * math.sqrt(ar)]
                    mean += [cx, cy, s_k_i / math.sqrt(ar), s_k_j * math.sqrt(ar)]

    # back to torch land
    output = torch.Tensor(mean).view(-1, 4)
    if clip:
        output.clamp_(max=1, min=0)
    return output


class SSD(nn.Module):

    def __init__(self, phase, nms_thresh=0.3, nms_conf_thresh=0.01):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = 2
        self.cfg = cfg

        resnet = torchvision.models.resnet152(pretrained=True)

        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)
        self.layer5 = nn.Sequential(
            *[nn.Conv2d(2048, 512, kernel_size=1),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True),
              nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True)]
        )
        self.layer6 = nn.Sequential(
            *[nn.Conv2d(512, 128, kernel_size=1, ),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
              nn.BatchNorm2d(256),
              nn.ReLU(inplace=True)]
        )

        output_channels = [256, 512, 1024, 2048, 512, 256]

        # FPN
        fpn_in = output_channels

        self.latlayer3 = nn.Conv2d(fpn_in[3], fpn_in[2], kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(fpn_in[2], fpn_in[1], kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(fpn_in[1], fpn_in[0], kernel_size=1, stride=1, padding=0)

        self.smooth3 = nn.Conv2d(fpn_in[2], fpn_in[2], kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Conv2d(fpn_in[1], fpn_in[1], kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Conv2d(fpn_in[0], fpn_in[0], kernel_size=1, stride=1, padding=0)

        # FEM
        cpm_in = output_channels

        self.cpm3_3 = FEM(cpm_in[0])
        self.cpm4_3 = FEM(cpm_in[1])
        self.cpm5_3 = FEM(cpm_in[2])
        self.cpm7 = FEM(cpm_in[3])
        self.cpm6_2 = FEM(cpm_in[4])
        self.cpm7_2 = FEM(cpm_in[5])

        # head
        head = pa_multibox(output_channels)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)

        if self.phase != 'onnx_export':
            self.detect = Detect(self.num_classes, 0, cfg['num_thresh'], nms_conf_thresh, nms_thresh,
                                 cfg['variance'])
            self.last_image_size = None
            self.last_feature_maps = None

        if self.phase == 'test':
            self.test_transform = TestBaseTransform((104, 117, 123))

    def forward(self, x):

        image_size = [x.shape[2], x.shape[3]]
        loc = list()
        conf = list()

        conv3_3_x = self.layer1(x)
        conv4_3_x = self.layer2(conv3_3_x)
        conv5_3_x = self.layer3(conv4_3_x)
        fc7_x = self.layer4(conv5_3_x)
        conv6_2_x = self.layer5(fc7_x)
        conv7_2_x = self.layer6(conv6_2_x)

        lfpn3 = upsample_product(self.latlayer3(fc7_x), self.smooth3(conv5_3_x))
        lfpn2 = upsample_product(self.latlayer2(lfpn3), self.smooth2(conv4_3_x))
        lfpn1 = upsample_product(self.latlayer1(lfpn2), self.smooth1(conv3_3_x))

        conv5_3_x = lfpn3
        conv4_3_x = lfpn2
        conv3_3_x = lfpn1

        sources = [conv3_3_x, conv4_3_x, conv5_3_x, fc7_x, conv6_2_x, conv7_2_x]

        sources[0] = self.cpm3_3(sources[0])
        sources[1] = self.cpm4_3(sources[1])
        sources[2] = self.cpm5_3(sources[2])
        sources[3] = self.cpm7(sources[3])
        sources[4] = self.cpm6_2(sources[4])
        sources[5] = self.cpm7_2(sources[5])

        # apply multibox head to source layers
        featuremap_size = []
        for (x, l, c) in zip(sources, self.loc, self.conf):
            featuremap_size.append([x.shape[2], x.shape[3]])
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            len_conf = len(conf)
            cls = mio_module(c(x), len_conf)
            conf.append(cls.permute(0, 2, 3, 1).contiguous())

        face_loc = torch.cat([o[:, :, :, :4].contiguous().view(o.size(0), -1) for o in loc], 1)
        face_loc = face_loc.view(face_loc.size(0), -1, 4)
        face_conf = torch.cat([o[:, :, :, :2].contiguous().view(o.size(0), -1) for o in conf], 1)
        face_conf = self.softmax(face_conf.view(face_conf.size(0), -1, self.num_classes))

        if self.phase != 'onnx_export':

            if self.last_image_size is None or self.last_image_size != image_size or self.last_feature_maps != featuremap_size:
                self.priors = get_prior_boxes(self.cfg, featuremap_size, image_size).to(face_loc.device)
                self.last_image_size = image_size
                self.last_feature_maps = featuremap_size
            with torch.no_grad():
                output = self.detect(face_loc, face_conf, self.priors)
        else:
            output = torch.cat((face_loc, face_conf), 2)
        return output

    def detect_on_image(self, source_image, target_size, device, is_pad=False, keep_thresh=0.3):

        image, shift_h_scaled, shift_w_scaled, scale = resize_image(source_image, target_size, is_pad=is_pad)

        x = torch.from_numpy(self.test_transform(image)).permute(2, 0, 1).to(device)
        x.unsqueeze_(0)

        detections = self.forward(x).cpu().numpy()

        scores = detections[0, 1, :, 0]
        keep_idxs = scores > keep_thresh  # find keeping indexes
        detections = detections[0, 1, keep_idxs, :]  # select detections over threshold
        detections = detections[:, [1, 2, 3, 4, 0]]  # reorder

        detections[:, [0, 2]] -= shift_w_scaled  # 0 or pad percent from left corner
        detections[:, [1, 3]] -= shift_h_scaled  # 0 or pad percent from top
        detections[:, :4] *= scale

        return detections