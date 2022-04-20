""" Utility script for downloading the FSGAN models.

This script should be placed in the root directory of the FSGAN repository.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from fsgan.utils.utils import download_from_url
import logging
import traceback


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--output', default='weights', metavar='DIR',
                    help='output directory')


model_links = {
    'https://drive.google.com/u/0/uc?id=1paSZfz-DG--huCK_Ts9dV9Y8DB8Wzn1T&export=download':
        'nfv_msrunet_256_1_2_reenactment_v2.1.pth',
    'https://drive.google.com/u/0/uc?id=1hC1dRKmn7Ux6aEDMA49ZXXyMY15Jyh8e&export=download':
        'ijbc_msrunet_256_1_2_inpainting_v2.pth',
    'https://drive.google.com/u/0/uc?id=1KoXPlYrVz5Z0h078VlKAnZ3ae9Sm8Mto&export=download':
        'ijbc_msrunet_256_1_2_blending_v2.pth',
    'https://drive.google.com/u/0/uc?id=1wVCHLFZG1SSfxdoAsiHQYc1zvOveyh-7&export=download':
        'celeba_unet_256_1_2_segmentation_v2.pth',
    'https://drive.google.com/u/0/uc?id=1wHR9ZwuN87KIILGwn2L5-oXh2NcxKMx4&export=download':
        'WIDERFace_DSFD_RES152.pth',
    'https://drive.google.com/u/0/uc?id=1toqwIO-syXaLQCvVhGRHHzc0TK7LaFz9&export=download':
        'hr18_wflw_landmarks.pth',
    'https://drive.google.com/u/0/uc?id=1f52AWf40YHd9l4OxxE7sugV8CNt1gY1m&export=download':
        'vggface2_vgg19_256_1_2_id.pth',
    'https://drive.google.com/u/0/uc?id=1shSuPCp9TZx0znSTUojg5tAvZAC1csUg&export=download':
        'celeba_vgg19_256_2_0_28_attr.pth',
    'https://drive.google.com/u/0/uc?id=1x0G8746WZbPK0BFLpX_IOOsUPLXWyh9c&export=download':
        'hopenet_robust_alpha1.pth'
}


def main(output='weights'):
    
    # Make sure the output directory exists
    os.makedirs(output, exist_ok=True)

    # For each mode link
    for i, (link, filename) in enumerate(model_links.items()):
        filename = os.path.split(link)[1] if filename is None else filename
        out_path = os.path.join(output, filename)
        if os.path.isfile(out_path):
            print('[%d/%d] Skipping "%s"' % (i + 1, len(model_links), filename))
            continue
        print('[%d/%d] Downloading "%s"...' % (i + 1, len(model_links), filename))
        try:
            download_from_url(link, out_path)
        except Exception as e:
            logging.error(traceback.format_exc())


if __name__ == "__main__":
    main(**vars(parser.parse_args()))
