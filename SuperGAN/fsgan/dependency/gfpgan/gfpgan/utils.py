from tkinter import Image
import cv2
import os
from cv2 import imshow
import torch
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from .archs.gfpgan_bilinear_arch import GFPGANBilinear
from .archs.gfpganv1_arch import GFPGANv1
from .archs.gfpganv1_clean_arch import GFPGANv1Clean

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))




class GFPGANer():
    """Helper for restoration with GFPGAN.

    It will detect and crop faces, and then resize the faces to 512x512.
    GFPGAN is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The GFPGAN architecture. Option: clean | original. Default: clean.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
    """

    def __init__(self, model_path, upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # initialize the GFP-GAN
        if arch == 'clean':
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'bilinear':
            self.gfpgan = GFPGANBilinear(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'original':
            self.gfpgan = GFPGANv1(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        # initialize face helper
        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device)

        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)
        

    @torch.no_grad()
    def enhance(self, imgs, has_aligned=False, only_center_face=False, paste_back=True):
        
        self.face_helper.clean_all()

        self.restored_faces = None
        
        # if has_aligned:  # the inputs are already aligned
        #     # img = cv2.resize(img, (512, 512))
        #     self.face_helper.cropped_faces = [img]
        # else:
        #     self.face_helper.read_image(img)
        #     # get face landmarks for each face
        #     self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
        #     # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
        #     # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
        #     # align and warp each face
        #     self.face_helper.align_warp_face()


        #>>>Edits
        data_dir = './data/gfpgan_test/'
        image_stages = {} 
        #>>> Edits end

        # face restoration
        for img in imgs:
            
            # prepare data
            cropped_face = F.resize(img, 512)
            self.face_helper.cropped_faces.append(img)

            #>>>Edits original changed commented
            cropped_face_t = cropped_face.unsqueeze(0).to(self.device)

            try:
                output = self.gfpgan(cropped_face_t, return_rgb=False)
                restored_face = F.resize(output[0].squeeze(0),256)
                stages = output[1]
            except RuntimeError as error:
                print(f'\tFailed inference for GFPGAN: {error}.')
                restored_face = cropped_face
            
            # restored_face = restored_face.astype('uint8')
            # self.restored_faces = torch.stack((self.restored_faces,restored_face))
            self.face_helper.add_restored_face(restored_face)
        
        #>>> Edits
                   
        image_stages["input"] = img
        image_stages["stages"] = stages
        image_stages["output"] = restored_face
        torch.save(image_stages, data_dir + 'gfpgan_image_stages.pth')        

        #>>> Edits end

        if not has_aligned and paste_back:
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
            else:
                bg_img = None

            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img
        else:
            #>>>Edits original changed commented
            # return self.face_helper.cropped_faces, self.face_helper.restored_faces, None
            return torch.stack(self.face_helper.restored_faces)