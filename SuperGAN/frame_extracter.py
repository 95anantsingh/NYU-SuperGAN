import cv2
from os.path import exists
from argparse import ArgumentParser

def extractImages(pathIn, pathOut):
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    print("\nInput Path: ", pathIn)
    print('\nExtracting')
    count = 0
    success = True
    if not exists(pathIn):
        print("Input path does not exists")
        exit()
    while success:
        cv2.imwrite(pathOut+"frame%d.jpg" % count, image)     
        success,image = vidcap.read()
        count += 1
    print('Done..\n')
    print("Output Path: ", pathOut,'\n')

if __name__=="__main__":
    a = ArgumentParser()
    a.add_argument("-i,","--pathIn", default="./FSGAN2/output/output2.mp4",help="path to video")
    a.add_argument("-o","--pathOut", default="./FSGAN2/output/output2/", help="path to images")
    args = a.parse_args()
    extractImages(args.pathIn, args.pathOut)