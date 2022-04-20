import os
from glob import glob
import pickle
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from face_detection_dsfd.face_ssd_infer import SSD
from face_detection_dsfd.data import widerface_640, TestBaseTransform


def parse_images(input, postfix='.jpg', indices=None):
    out_dir = None
    if len(input) == 1:
        assert os.path.isfile(input[0]) or os.path.isdir(input[0]), 'input must be a file or a directory: "%s"' % input
        if os.path.isfile(input[0]):
            img_paths = input
            out_dir = os.path.split(input[0])[0]
        elif os.path.isdir(input[0]):
            img_paths = sorted(glob(os.path.join(input[0], '*' + postfix)))
            out_dir = input[0]
    elif len(input) == 2:
        list_file_path = os.path.join(input[0], input[1]) if not os.path.isfile(input[1]) else input[1]
        if os.path.isdir(input[0]) and os.path.isfile(list_file_path):    # Directory and list file
            rel_paths = np.loadtxt(list_file_path, str)
            img_paths = [os.path.join(input[0], p) for p in rel_paths]
            out_dir = input[0]

    img_paths = eval('img_paths[%s]' % indices) if indices is not None else img_paths

    return img_paths, out_dir


def main(input, out_dir=None, indices=None, detection_model_path='weights/WIDERFace_DSFD_RES152.pth', postfix='.jpg',
         out_postfix='_dsfd.pkl', image_padding=None, max_res=2048, display=False):
    # Parse images
    img_paths, suggested_out_dir = parse_images(input, postfix, indices)
    # out_dir = suggested_out_dir if out_dir is None else out_dir

    # Initialize device
    cuda = True
    torch.set_grad_enabled(False)
    device = torch.device('cuda:{}'.format(0))
    if cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Initialize detection model
    net = SSD("test")
    net.load_state_dict(torch.load(detection_model_path))
    net.eval()

    transform = TestBaseTransform((104, 117, 123))
    cfg = widerface_640
    thresh = cfg['conf_thresh']

    # For each image file
    for n, img_path in enumerate(img_paths):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        if out_dir is None:
            curr_cache_path = os.path.splitext(img_path)[0] + out_postfix
        else:
            curr_cache_path = os.path.join(out_dir, img_name + out_postfix)

        if os.path.exists(curr_cache_path):
            print('[%d/%d] Skipping "%s"' % (n + 1, len(img_paths), img_name))
            continue
        else:
            print('[%d/%d] Processing "%s"...' % (n + 1, len(img_paths), img_name))

        # Process image
        img = cv2.imread(img_path)
        image_size = img.shape[:2]

        # Scale images above the maximum resolution
        scale = np.array(image_size[::-1] * 2, dtype=float)
        shrink = min(max_res / np.max(np.array(image_size)), 1.0)
        if shrink < 1.0:
            img_scaled = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
            image_size = img_scaled.shape[:2]
            frame_tensor = torch.from_numpy(transform(img_scaled)[0]).permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            frame_tensor = torch.from_numpy(transform(img)[0]).permute(2, 0, 1).unsqueeze(0).to(device)

        # Pad image
        if image_padding is not None:
            image_pad_size = np.round(np.array(image_size[::-1]) * image_padding).astype(int)
            frame_tensor = F.pad(frame_tensor, [image_pad_size[0], image_pad_size[0],
                                                image_pad_size[1], image_pad_size[1]], 'reflect')
            image_pad_size = np.round(np.array(img.shape[1::-1]) * image_padding).astype(int)
            image_size = np.round(np.array(img.shape[1::-1]) * (1 + image_padding * 2)).astype(int)
            scale = np.array(tuple(image_size) * 2, dtype=float)

        # Detect faces
        detections = net(frame_tensor)

        det = []
        # shrink = 1.0
        # scale = torch.Tensor([image_size[1] / shrink, image_size[0] / shrink,
        #                       image_size[1] / shrink, image_size[0] / shrink])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                curr_det = detections[0, i, j, [1, 2, 3, 4, 0]].cpu().numpy()
                # curr_det[:4] *= scale.cpu().numpy()
                curr_det[:4] *= scale
                det.append(curr_det)
                j += 1

        if len(det) == 0:
            det = np.array([], dtype='float32')
        else:
            det = np.row_stack((det))
            det = det[det[:, 4] > 0.5, :4]

        # Restore detection relative to original image
        if image_padding is not None:
            det[:, :2] -= image_pad_size
            det[:, 2:] -= image_pad_size

        # Render
        if display:
            det_display = np.round(det).astype(int)
            render_img = img
            if shrink < 1.0:
                render_img = cv2.resize(render_img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
                det_display = (det_display * shrink).astype('float32')
            for rect in det_display:
                cv2.rectangle(render_img, tuple(rect[:2]), tuple(rect[2:]), (0, 0, 255), 1)

            cv2.imshow('render_img', render_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Write detection to file
        with open(curr_cache_path, 'wb') as f:
            pickle.dump([det], f)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser(os.path.splitext(os.path.basename(__file__))[0])
    parser.add_argument('input', metavar='PATH', nargs='+',
                        help='input paths')
    parser.add_argument('-o', '--output', metavar='DIR',
                        help='output directory')
    parser.add_argument('-i', '--indices', default=None,
                        help='python style indices (e.g 0:10')
    parser.add_argument('-dm', '--detection_model', metavar='PATH', default='weights/WIDERFace_DSFD_RES152.pth',
                        help='path to face detection model')
    parser.add_argument('-p', '--postfix', default='.jpg', metavar='POSTFIX',
                        help='input image postfix')
    parser.add_argument('-op', '--out_postfix', default='_dsfd.pkl', metavar='POSTFIX',
                        help='output file postfix')
    parser.add_argument('-ip', '--image_padding', type=float, metavar='F',
                        help='image padding relative to image size')
    parser.add_argument('-mr', '--max_res', default=2048, type=int, metavar='N',
                        help='maximum detection resolution (higher resolution images will be scaled down)')
    parser.add_argument('-d', '--display', action='store_true',
                        help='display the rendering')
    args = parser.parse_args()
    main(args.input, args.output, args.indices, args.detection_model, args.postfix, args.out_postfix,
         args.image_padding, args.max_res, args.display)
