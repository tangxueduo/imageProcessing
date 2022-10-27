import cv2

# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture("Resources/Cars.mp4")
if vid_capture.isOpened() == False:
    print("can not opening the video file")
else:
    fps = int(vid_capture.get(5))
