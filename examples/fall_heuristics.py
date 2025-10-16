import numpy as np
from math import atan2, degrees





NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12



## this function is just for detecting body parts confidently
"""
Keypoints are 17 points from yolo, with a matching index to that body part.

x, y = location of the body part in the video frame

"""
def _pt(keypoints, index, min_confidence=.5):
    x, y, confidence = keypoints[index]
    
    if confidence < min_confidence:
        return None
    return (x, y)


## this extracts 4 points because a torso is a straight line between shoulders and hips
def torso_angle_degree(k):
    ls = _pt(k, LEFT_SHOULDER)
    rs = _pt(k, RIGHT_SHOULDER)
    lh = _pt(k, LEFT_HIP)
    rh = _pt(k, RIGHT_HIP)
    
    if not (ls and rs and lh and rh):
        return None
    
    mid_shoulder = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
    mid_hip = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
    
    dx = mid_hip[0] - mid_shoulder[0]
    dy = mid_hip[1] - mid_shoulder[1]
    
    angle = abs(degrees(atan2(dx, dy)))
    return angle


def bbox_aspect_ratio(xyxy):
    x1, y1, x2, y2 = xyxy
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return height / width


def head_vs_hip(keypoints):
    ns = _pt(keypoints, NOSE)
    rh = _pt(keypoints, RIGHT_HIP)
    lh = _pt(keypoints, LEFT_HIP)
    
    if ns is None or lh is None or rh is None:
        return 0
    hip_height = (lh[1] + rh[1]) / 2

    return 1 if ns[1] > hip_height else 0


def fall_score(kp, bbox):
    
    score = 0
    
    torso_angle = torso_angle_degree(kp) or 0
    bounding_box_ratio = bbox_aspect_ratio(bbox) or 1
    head_hip = head_vs_hip(kp) or 0
    
    
    if torso_angle > 45:
        score += .5
    if bounding_box_ratio < 0.6:
        score += .3
    if head_hip == 1:
        score += .2
        
    return min(score, 1)
    
    

    
    









