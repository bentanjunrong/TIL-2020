import cv2
import numpy

def crop_frame_by(frame,crop_by_factor,crop_bot=False):
    crop_by = crop_by_factor #lol pls forgive me for this
    width = (1280/crop_by)/2
    x_left = int((640) - width) # from center
    x_right = int((640) + width) #from center
    if crop_bot: return frame[0:625,x_left:x_right]
    else: return frame[0:720,x_left:x_right]
