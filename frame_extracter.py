import cv2
from argparse import ArgumentParser

def extractImages(pathIn, pathOut):
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    print("\nInput Path: ", pathIn)
    print('\nExtracting')
    count = 0
    success = True
    while success:
        cv2.imwrite(pathOut+"frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
    print('Done..\n')
    print("Output Path: ", pathOut,'\n')

if __name__=="__main__":
    a = ArgumentParser()
    a.add_argument("-i,","--pathIn", default="./FSGAN2/output/output.mp4",help="path to video")
    a.add_argument("-o","--pathOut", default="./FSGAN2/output/frames/", help="path to images")
    args = a.parse_args()
    extractImages(args.pathIn, args.pathOut)