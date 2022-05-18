import os
import ffmpeg
from fsgan.inference.swap import FaceSwapping
from fsgan.criterions.vgg_loss import VGGLoss

# Utility functions
def encode_audio(video_path, audio_path, output_path):
  ffmpeg.concat(ffmpeg.input(video_path), ffmpeg.input(audio_path), v=1, a=1) \
    .output(output_path, strict='-2').run(overwrite_output=True)


# Number of finetune iterations on the source subject: min:100, max:2000, step:1
finetune_iterations = 1000

# If True, the inner part of the mouth will be removed from the segmentation:
seg_remove_mouth = True

# Segmentation batch size: min:1, max:64, step:1
seg_batch_size = 64

# Inference batch size: min:1, max:64, step:1
batch_size = 8

# Set Data Directory
data_dir = "./data/"

# Path to the weights directory (make sure it is correct):
weights_dir = data_dir + '/weights'

# Load model weights
detection_model = os.path.join(weights_dir,'WIDERFace_DSFD_RES152.pth')
pose_model = os.path.join(weights_dir, 'hopenet_robust_alpha1.pth')
lms_model = os.path.join(weights_dir, 'hr18_wflw_landmarks.pth')
seg_model = os.path.join(weights_dir, 'celeba_unet_256_1_2_segmentation_v2.pth')
reenactment_model = os.path.join(weights_dir, 'nfv_msrunet_256_1_2_reenactment_v2.1.pth')
completion_model = os.path.join(weights_dir, 'ijbc_msrunet_256_1_2_inpainting_v2.pth')
blending_model = os.path.join(weights_dir, 'ijbc_msrunet_256_1_2_blending_v2.pth')
criterion_id_path = os.path.join(weights_dir, 'vggface2_vgg19_256_1_2_id.pth')
criterion_id = VGGLoss(criterion_id_path)

# Initial Swapping object
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
source_path = data_dir + 'input/shinzo_abe.mp4'

# Source selection method ["longest" | sequence number]:
select_source = "longest"

# Target path
target_path = data_dir + 'input/conan_obrien.mp4'

# Target selection method
select_target = 'longest'

# Outputs
output_tmp_path = data_dir + 'output/output_tmp.mp4'
output_path = data_dir + 'output/output.mp4'

# Select upscale value for GFPGAN
# upscale = 2

face_swapping(source_path, target_path, output_tmp_path,
              select_source, select_target, finetune)

# Encode with audio and save result
encode_audio(output_tmp_path, target_path, output_path)
os.remove(output_tmp_path)

