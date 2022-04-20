import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from face_ssd_infer import SSD
from face_ssd import build_ssd
from data import widerface_640, TestBaseTransform


def main(input_path, output_path):

    # Create model
    # net = SSD("onnx_export")
    # net.load_state_dict(torch.load(input_path))
    # net.eval()

    # Initialize detection model
    cfg = widerface_640
    thresh = cfg['conf_thresh']
    net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])  # initialize SSD
    net.load_state_dict(torch.load(input_path))
    net.eval()
    print('Finished loading detection model!')

    # Generate a torch.jit.ScriptModule via tracing
    print('=> tracing module...')
    input = torch.rand(1, 3, 720, 1280)
    traced_script_module = torch.jit.trace(net, input)

    # Serialize script module
    # output_path = os.path.join(exp_dir, 'unet_face_segmentation_256.pt')
    print("=> saving script module to '{}'".format(output_path))
    traced_script_module.save(output_path)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('Export')
    parser.add_argument('input',
                        help='path to experiment directory')
    parser.add_argument('-o', '--output', default='weights/WIDERFace_DSFD_RES152.pt', metavar='PATH',
                        help='output path')
    args = parser.parse_args()
    main(args.input, args.output)