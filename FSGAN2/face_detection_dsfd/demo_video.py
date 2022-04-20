import os
from tqdm import tqdm
import numpy as np
import cv2
import torch
from face_ssd import build_ssd
from data import widerface_640, TestBaseTransform


def main(input_path, output_path, detection_model_path):
    cuda = True
    torch.set_grad_enabled(False)
    if cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Initialize detection model
    cfg = widerface_640
    thresh = cfg['conf_thresh']
    net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])  # initialize SSD
    net.load_state_dict(torch.load(detection_model_path))
    net = net.cuda()
    net.eval()
    print('Finished loading detection model!')

    transform = TestBaseTransform((104, 117, 123))

    # Open target video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError('Failed to read video: ' + input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if frame is None:
            continue

        # Process
        det0 = infer(net, frame, transform, thresh, True, shrink)
        det1 = infer_flip(net, frame, transform, thresh, True, shrink)
        # shrink detecting and shrink only detect big face
        st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
        det_s = infer(net, frame, transform, thresh, True, st)
        index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
        det_s = det_s[index, :]

        det = np.row_stack((det0, det1, det_s))
        det = bbox_vote(det.astype(float))

        det = np.round(det[det[:, 4] > 0.5, :4]).astype(int)


        # Render
        render_img = frame
        for rect in det:
            # cv2.rectangle(render_img, tuple(rect[:2]), tuple(rect[:2] + rect[2:]), (0, 0, 255), 1)
            cv2.rectangle(render_img, tuple(rect[:2]), tuple(rect[2:]), (0, 0, 255), 1)
        if out_vid is not None:
            out_vid.write(render_img)
        cv2.imshow('render_img', render_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


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
    parser.add_argument('-dm', '--detection_model', metavar='PATH', default='weights/WIDERFace_DSFD_RES152.pth',
                        help='path to face detection model')
    args = parser.parse_args()
    main(args.input, args.output, args.detection_model)