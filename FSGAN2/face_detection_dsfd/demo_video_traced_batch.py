import os
import math
from tqdm import tqdm
import numpy as np
import cv2
import torch
from face_ssd import build_ssd
from data import widerface_640, TestBaseTransform
from layers.functions.detection import Detect


def main(input_path, output_path, detection_model_path='weights/WIDERFace_DSFD_RES152.pt', batch_size=8, display=False):
    cuda = True
    torch.set_grad_enabled(False)
    device = torch.device('cuda:{}'.format(0))
    if cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Initialize detection model
    # cfg = widerface_640
    # thresh = cfg['conf_thresh']
    # net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])  # initialize SSD
    # net.load_state_dict(torch.load(detection_model_path))
    # net = net.cuda()
    # net.eval()

    cfg = widerface_640
    thresh = cfg['conf_thresh']
    net = torch.jit.load(detection_model_path, map_location=device)
    net.eval()
    print('Finished loading detection model!')

    transform = TestBaseTransform((104, 117, 123))
    detect = Detect(cfg['num_classes'], 0, cfg['num_thresh'], cfg['conf_thresh'], cfg['nms_thresh'])

    # Open target video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError('Failed to read video: ' + input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate priors
    image_size = (target_vid_height, target_vid_width)
    featuremap_size = [(math.ceil(image_size[0] / (2 ** (2 + i))), math.ceil(image_size[1] / (2 ** (2 + i))))
                       for i in range(6)]
    priors = get_prior_boxes(cfg, featuremap_size, image_size).to(device)

    # Initialize output video file
    if output_path is not None:
        if os.path.isdir(output_path):
            output_filename = os.path.splitext(os.path.basename(input_path))[0] + '.mp4'
            output_path = os.path.join(output_path, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'x264')
        out_vid = cv2.VideoWriter(output_path, fourcc, fps, (target_vid_width, target_vid_height))
    else:
        out_vid = None

    #
    max_im_shrink = ((2000.0 * 2000.0) / (target_vid_height * target_vid_width)) ** 0.5
    shrink = max_im_shrink if max_im_shrink < 1 else 1

    # For each frame in the video
    frame_bgr_list = []
    frame_tensor_list = []
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if frame is None:
            continue

        # Gather batches
        frame_bgr_list.append(frame)
        frame_tensor = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1).unsqueeze(0).to(device)
        frame_tensor_list.append(frame_tensor)
        if len(frame_tensor_list) < batch_size and (i + 1) < total_frames:
            continue
        frame_tensor_batch = torch.cat(frame_tensor_list, dim=0)

        # Process
        pred = net(frame_tensor_batch)
        detections_batch = detect(pred[:, :, :4], pred[:, :, 4:], priors)
        for b, detections in enumerate(detections_batch):
            detections = detections.unsqueeze(0)

            det = []
            shrink = 1.0
            scale = torch.Tensor([image_size[1] / shrink, image_size[0] / shrink,
                                  image_size[1] / shrink, image_size[0] / shrink])
            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= thresh:
                    curr_det = detections[0, i, j, [1, 2, 3, 4, 0]].cpu().numpy()
                    curr_det[:4] *= scale.cpu().numpy()
                    det.append(curr_det)
                    j += 1

            det = np.row_stack((det))
            # if det.shape[0] > 1:
            #     det = bbox_vote(det.astype(float))
            det = np.round(det[det[:, 4] > 0.5, :4]).astype(int)

            # Render
            if display:
                render_img = frame_bgr_list[b]
                for rect in det:
                    # cv2.rectangle(render_img, tuple(rect[:2]), tuple(rect[:2] + rect[2:]), (0, 0, 255), 1)
                    cv2.rectangle(render_img, tuple(rect[:2]), tuple(rect[2:]), (0, 0, 255), 1)
                if out_vid is not None:
                    out_vid.write(render_img)
                cv2.imshow('render_img', render_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Clear lists
        frame_bgr_list.clear()
        frame_tensor_list.clear()


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


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)
        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    dets = dets[0:750, :]
    return dets


def infer(net , img , transform , thresh , cuda , shrink):
    if shrink != 1:
        img = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    # x = Variable(x.unsqueeze(0) , volatile=True)
    x = x.unsqueeze(0)
    if cuda:
        x = x.cuda()
    #print (shrink , x.shape)
    y = net(x)      # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([ img.shape[1]/shrink, img.shape[0]/shrink,
                         img.shape[1]/shrink, img.shape[0]/shrink] )
    det = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            #label_name = labelmap[i-1]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3])
            det.append([pt[0], pt[1], pt[2], pt[3], score])
            j += 1
    if (len(det)) == 0:
        det = [ [0.1,0.1,0.2,0.2,0.01] ]
    det = np.array(det)

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det

def infer_flip(net , img , transform , thresh , cuda , shrink):
    img = cv2.flip(img, 1)
    det = infer(net , img , transform , thresh , cuda , shrink)
    det_t = np.zeros(det.shape)
    det_t[:, 0] = img.shape[1] - det[:, 2]
    det_t[:, 1] = det[:, 1]
    det_t[:, 2] = img.shape[1] - det[:, 0]
    det_t[:, 3] = det[:, 3]
    det_t[:, 4] = det[:, 4]
    return det_t


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('demo_video')
    parser.add_argument('input', metavar='VIDEO',
                        help='path to input video')
    parser.add_argument('-o', '--output', default=None, metavar='DIR',
                        help='output directory')
    parser.add_argument('-dm', '--detection_model', metavar='PATH', default='weights/WIDERFace_DSFD_RES152.pt',
                        help='path to face detection model')
    parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N',
                        help='batch size (default: 8)')
    parser.add_argument('-d', '--display', action='store_true',
                        help='display the rendering')
    args = parser.parse_args()
    main(args.input, args.output, args.detection_model, args.batch_size, args.display)
