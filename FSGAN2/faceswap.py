import os
import ffmpeg
from base64 import b64encode
from IPython.display import HTML, clear_output
from fsgan.inference.swap import FaceSwapping
from fsgan.criterions.vgg_loss import VGGLoss

# Utility functions
def encode_audio(video_path, audio_path, output_path):
  ffmpeg.concat(ffmpeg.input(video_path), ffmpeg.input(audio_path), v=1, a=1) \
    .output(output_path, strict='-2').run(overwrite_output=True)


def display_video(video_path, width=640, clear=True):
  vid_data = open(video_path,'rb').read()
  vid_url = 'data:video/mp4;base64,' + b64encode(vid_data).decode()

  if clear:
    clear_output()

  html = HTML(f"""
  <video width={width} controls>
    <source src={vid_url} type="video/mp4">
  </video>
  """)
  print(html)
  return html


# Path to the weights directory (make sure it is correct):
weights_dir = './weights'

# Number of finetune iterations on the source subject: min:100, max:2000, step:1
finetune_iterations = 1000

# If True, the inner part of the mouth will be removed from the segmentation:
seg_remove_mouth = True

# Segmentation batch size: min:1, max:64, step:1
seg_batch_size = 64

# Inference batch size: min:1, max:64, step:1
batch_size = 8


# /home/as14229/NYU_HPC/GitHub/NYU-SuperGAN/FSGAN2/face_detection_dsfd/face_ssd_infer.py:239: 
# UserWarning: An output with one or more elements was resized since it had shape [107], which 
# does not match the required output shape [98].This behavior is deprecated, and in a future 
# PyTorch release outputs will not be resized unless they have zero elements. 
# You can explicitly reuse an out tensor t by resizing it, inplace, to zero elements with t.resize_(0). 
# (Triggered internally at  /opt/conda/conda-bld/pytorch_1640811803361/work/aten/src/ATen/native/Resize.cpp:23.)
# torch.index_select(y2, 0, idx, out=yy2)

data_dir = "/home/as14229/Shared/SuperGAN/data/"


detection_model = os.path.join(weights_dir, 'v2/WIDERFace_DSFD_RES152.pth')
pose_model = os.path.join(weights_dir, 'shared/hopenet_robust_alpha1.pth')
lms_model = os.path.join(weights_dir, 'v2/hr18_wflw_landmarks.pth')
seg_model = os.path.join(weights_dir, 'v2/celeba_unet_256_1_2_segmentation_v2.pth')
reenactment_model = os.path.join(weights_dir, 'v2/nfv_msrunet_256_1_2_reenactment_v2.1.pth')
completion_model = os.path.join(weights_dir, 'v2/ijbc_msrunet_256_1_2_inpainting_v2.pth')
blending_model = os.path.join(weights_dir, 'v2/ijbc_msrunet_256_1_2_blending_v2.pth')
criterion_id_path = os.path.join(weights_dir, 'v2/vggface2_vgg19_256_1_2_id.pth')
criterion_id = VGGLoss(criterion_id_path)


face_swapping = FaceSwapping(
    detection_model=detection_model, pose_model=pose_model, lms_model=lms_model,
    seg_model=seg_model, reenactment_model=reenactment_model,
    completion_model=completion_model, blending_model=blending_model,
    criterion_id=criterion_id,
    finetune=True, finetune_save=True, finetune_iterations=finetune_iterations,
    seg_remove_mouth=finetune_iterations, batch_size=batch_size,
    seg_batch_size=seg_batch_size, encoder_codec='mp4v')


# Face Swapping
# Toggle whether to finetune the reenactment generator:
finetune = True

# Source path
source_path = '../data/input/elon_musk.mp4'

# Source selection method ["longest" | sequence number]:
select_source = "11"

# Target path
target_path = '../data/input/conan_obrien.mp4'

# Target selection method
select_target = 'longest'

output_tmp_path = './output/output_tmp.mp4'
output_path = './output/output3.mp4'

face_swapping(source_path, target_path, output_tmp_path,
              select_source, select_target, finetune)

# Encode with audio and display result
encode_audio(output_tmp_path, target_path, output_path)
os.remove(output_tmp_path)
# display_video(output_path)
