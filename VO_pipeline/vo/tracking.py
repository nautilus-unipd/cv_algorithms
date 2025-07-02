import numpy as np
import cv2


def extract_features(img1,detector_name= "orb"):
    #esiste la possibiltà di passare parametri personalizzati alla funzione create()
    #in base a risultati, verrà considerata questa opzione, previa scelte adeguate e documentate
    if detector_name== "orb":
        detector= cv2.ORB.create()
    elif detector_name=="sift":
        detector= cv2.SIFT.create()
    else:
        print("Not valid detector")
    
    kp,des= detector.detectAndCompute(img1, mask=None)
    
    return kp, des


def match_features(des1,des2, matcher="BF", detector="orb"):
    if matcher== "BF":
        if detector == "orb":
            bf = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=False)
        elif detector== "sift":
            bf = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=False)
        else:
            print("Not valid detector")
    else:
        print("Not valid matcher")
        
    matches= bf.knnMatch(des1,des2, k=2)
    
    #VALUTAZIONE UTILIZZO FILTRO EPIPOLARE
    return matches
        
#ratio test
def filter_matches(matches,threshold):
    filtered_matches=[]
    for x, y in matches:
        if x.distance <= y.distance*threshold:
            filtered_matches.append(x)
    return filtered_matches