# importing libraries
import os
import cv2 
from argparse import ArgumentParser


num_of_images = len(os.listdir(pathIn+'.'))

# Video Generating function
def generate_video():
    image_folder = '.' # make sure to use your folder
    video_name = 'mygeneratedvideo.avi'
      
    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")]
     
  
    frame = cv2.imread(os.path.join(image_folder, images[0]))
  
    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape  
  
    video = cv2.VideoWriter(video_name, 0, 1, (width, height)) 
  
    # Appending the images to the video one by one
    for image in images: 
        video.write(cv2.imread(os.path.join(image_folder, image))) 
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated
  
  
# Calling the generate_video function
if __name__=="__main__":
    a = ArgumentParser()
    a.add_argument("-i,","--pathIn", default="./FSGAN2/output/output.mp4",help="path to video")
    a.add_argument("-o","--pathOut", default="./FSGAN2/output/frames/", help="path to images")
    args = a.parse_args()
    generate_video(args.pathIn, args.pathOut)