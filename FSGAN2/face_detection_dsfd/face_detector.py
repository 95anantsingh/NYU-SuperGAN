import os
import math
import pickle
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
# from face_ssd import build_ssd
from face_detection_dsfd.face_ssd_infer import SSD
from face_detection_dsfd.data import widerface_640, TestBaseTransform
from face_detection_dsfd.layers.functions.detection import Detect


class FaceDetector(object):
    def __init__(self, out_postfix='_dsfd.pkl', detection_model_path='weights/WIDERFace_DSFD_RES152.pth',
                 gpus=None, batch_size=8, verbose=0):
        super(FaceDetector, self).__init__()
        # assert torch.cuda.is_available(), 'CUDA must be available!'
        self.out_postfix = out_postfix
        self.batch_size = batch_size
        self.verbose = verbose

        # Set default tensor type
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # Initialize device
        torch.set_grad_enabled(False)
        self.device, self.gpus = set_device(gpus, silence=True)

        # Initialize detection model
        self.net = SSD("test").to(self.device)
        self.net.load_state_dict(torch.load(detection_model_path))
        self.net.eval()

        # Initialize configuration
        self.transform = TestBaseTransform((104, 117, 123))
        self.cfg = widerface_640
        self.thresh = self.cfg['conf_thresh']

        # Support multiple GPUs
        if self.gpus and len(self.gpus) > 1:
            self.net = nn.DataParallel(self.net, self.gpus)

        # Reset default tensor type
        # torch.set_default_tensor_type('torch.FloatTensor')

    def __call__(self, input_path, output_path=None):
        if output_path is None:
            output_filename = os.path.splitext(os.path.basename(input_path))[0] + self.out_postfix
            output_dir = os.path.split(input_path)[0]
            output_path = os.path.join(output_dir, output_filename)
        elif os.path.isdir(output_path):
            output_filename = os.path.splitext(os.path.basename(input_path))[0] + self.out_postfix
            output_path = os.path.join(output_path, output_filename)

        if os.path.isfile(output_path):
            return
        print('=> Detecting faces in video: "%s..."' % os.path.basename(input_path))

        # Open input video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError('Failed to read video: ' + input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        input_vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        image_size = (input_vid_height, input_vid_width)

        # Set default tensor type
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # For each frame in the video
        frame_bgr_list = []
        frame_tensor_list = []
        det_list = []
        for i in tqdm(range(total_frames), unit='frames'):
            ret, frame = cap.read()
            if frame is None:
                continue

            # Gather batches
            frame_bgr_list.append(frame)
            frame_tensor = torch.from_numpy(self.transform(frame)[0]).permute(2, 0, 1).unsqueeze(0).to(self.device)
            frame_tensor_list.append(frame_tensor)
            if len(frame_tensor_list) < self.batch_size and (i + 1) < total_frames:
                continue
            frame_tensor_batch = torch.cat(frame_tensor_list, dim=0)

            # Process
            detections_batch = self.net(frame_tensor_batch)
            # detections_batch = detect(pred[:, :, :4], pred[:, :, 4:], priors)
            for b, detections in enumerate(detections_batch):
                detections = detections.unsqueeze(0)

                det = []
                shrink = 1.0
                scale = torch.Tensor([image_size[1] / shrink, image_size[0] / shrink,
                                      image_size[1] / shrink, image_size[0] / shrink])
                for i in range(detections.size(1)):
                    j = 0
                    while detections[0, i, j, 0] >= self.thresh:
                        curr_det = detections[0, i, j, [1, 2, 3, 4, 0]].cpu().numpy()
                        curr_det[:4] *= scale.cpu().numpy()
                        det.append(curr_det)
                        j += 1

                if len(det) == 0:
                    det_list.append(np.array([], dtype='float32'))
                else:
                    det = np.row_stack((det))
                    # if det.shape[0] > 1:
                    #     det = bbox_vote(det.astype(float))
                    det_filtered = det[det[:, 4] > 0.5, :4]
                    det_list.append(det_filtered)

                # Render
                if self.verbose > 0:
                    det_display = np.round(det_filtered).astype(int)
                    render_img = frame_bgr_list[b]
                    for rect in det_display:
                        # cv2.rectangle(render_img, tuple(rect[:2]), tuple(rect[:2] + rect[2:]), (0, 0, 255), 1)
                        cv2.rectangle(render_img, tuple(rect[:2]), tuple(rect[2:]), (0, 0, 255), 1)
                    # if out_vid is not None:
                    #     out_vid.write(render_img)
                    cv2.imshow('render_img', render_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            # Clear lists
            frame_bgr_list.clear()
            frame_tensor_list.clear()

        # Write to file
        with open(output_path, 'wb') as f:
            pickle.dump(det_list, f)

        # Reset default tensor type
        torch.set_default_tensor_type('torch.FloatTensor')

    def detect(self, frame_bgr):
        frame_tensor = torch.from_numpy(self.transform(frame_bgr)[0]).permute(2, 0, 1).unsqueeze(0).to(self.device)
        detections = self.net(frame_tensor)[0].unsqueeze(0)

        det = []
        shrink = 1.0
        scale = torch.Tensor([frame_bgr.shape[1] / shrink, frame_bgr.shape[0] / shrink,
                              frame_bgr.shape[1] / shrink, frame_bgr.shape[0] / shrink])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= self.thresh:
                curr_det = detections[0, i, j, [1, 2, 3, 4, 0]].cpu().numpy()
                curr_det[:4] *= scale.cpu().numpy()
                det.append(curr_det)
                j += 1

        if len(det) == 0:
            return np.array([], dtype='float32')

        det = np.row_stack((det))
        det_filtered = det[det[:, 4] > 0.5, :4]

        return det_filtered


def set_device(gpus=None, use_cuda=True, silence=False):
    use_cuda = torch.cuda.is_available() if use_cuda else use_cuda
    if use_cuda:
        gpus = list(range(torch.cuda.device_count())) if not gpus else gpus
        if not silence:
            print('=> using GPU devices: {}'.format(', '.join(map(str, gpus))))
    else:
        gpus = None
        if not silence:
            print('=> using CPU device')
    device = torch.device('cuda:{}'.format(gpus[0])) if gpus else torch.device('cpu')

    return device, gpus


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


def main(input_path, output_path, detection_model_path='weights/WIDERFace_DSFD_RES152.pth', batch_size=8,
         display=False, out_postfix='_dsfd.pkl', gpus=None):
    face_detector = FaceDetector(out_postfix, detection_model_path, gpus, batch_size, display)
    face_detector(input_path, output_path)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('face_detector')
    parser.add_argument('input', metavar='VIDEO',
                        help='path to input video')
    parser.add_argument('-o', '--output', default=None, metavar='PATH',
                        help='output directory')
    parser.add_argument('-dm', '--detection_model', metavar='PATH', default='weights/WIDERFace_DSFD_RES152.pth',
                        help='path to face detection model')
    parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N',
                        help='batch size (default: 8)')
    parser.add_argument('-d', '--display', action='store_true',
                        help='display the rendering')
    parser.add_argument('-op', '--out_postfix', default='_dsfd.pkl', metavar='POSTFIX',
                        help='output file postfix')
    parser.add_argument('--gpus', nargs='+', type=int, metavar='N',
                        help='list of gpu ids to use (default: all)')
    args = parser.parse_args()
    main(args.input, args.output, args.detection_model, args.batch_size, args.display, args.out_postfix, args.gpus)
