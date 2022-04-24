# importing libraries
import os
from argparse import ArgumentParser

# Video Generating function
def generate_video(pathInV, pathInA, pathOut, fps):
    image_dir = pathInV
    audio_path = pathInA
    video_path = pathOut
    
    command = f'ffmpeg -y -framerate {fps} -i {image_dir}/frame%d.jpg -i {audio_path} \
                -map 0:v -c:v libx264 -map 1:a -c:a copy -r {fps} {video_path}'

    os.system(r'{}'.format(command))
  
# Calling the generate_video function
if __name__=="__main__":
    
    # Set Data Directory
    data_dir = "/home/as14229/Shared/SuperGAN/data/"

    a = ArgumentParser()
    a.add_argument("-i,","--pathInFrames", default=data_dir+'fsgan_outputs/output1_2/restored_imgs',help="Path to frames")
    a.add_argument("-a","--pathInAudio", default=data_dir+'input/conan_obrien.mp4', help="path to audio")
    a.add_argument("-o","--pathOut", default=data_dir+'/output/output1_2.mp4', help="path to video")
    a.add_argument("-f","--fps", default=30, help="FPS of video")

    args = a.parse_args()
    generate_video(args.pathInFrames, args.pathInAudio, args.pathOut, args.fps)