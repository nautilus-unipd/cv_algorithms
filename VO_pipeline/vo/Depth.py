import numpy as np
import cv2

def preprocessing(img):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(img)
    
    process_image = cv2.GaussianBlur(clahe_img, (3,3), 0)
    return process_image


def compute_disparity_map(img_left,img_right,matcher_name="bm"):
    #SCELTA PROGETTUALE: inizialmente verificare risultati con orb anche se quest'ultimo soffre molto in presenza
    # di scene omogenee, come fondale marino. Post-risultati si valuterà il passaggio a sgbm (più costoso e pesante)
    #ma più affidabile
    sizeWindow= 6
    blockSize=11
    numDisparities= sizeWindow*16
    min_disp=0

    if matcher_name == "bm":
        matcher= cv2.StereoBM.create(numDisparities= numDisparities,
                                    blockSize=blockSize)
    elif matcher_name=="sgbm":
        
        matcher = cv2.StereoSGBM_create(
                                            minDisparity=min_disp,
                                            numDisparities=numDisparities,
                                            blockSize=sizeWindow,
                                            P1=8 * 3 * sizeWindow**2,
                                            P2=32 * 3 * sizeWindow**2,
                                            disp12MaxDiff=1,
                                            uniquenessRatio=10,
                                            speckleWindowSize=100,
                                            speckleRange=32,
                                            preFilterCap=63,
                                            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                            )
    else:
        print("Not valid Matcher")

    assert len(img_left.shape) == 2, "left_image not in grayscale"
    assert len(img_right.shape) == 2, "right_image not in grayscale"

    img_left=preprocessing(img_left)
    img_right=preprocessing(img_right)
        
    #compute disparity map
    disparity_map= matcher.compute(img_left,img_right).astype(np.float32)/16

    #filtro mediano per rimuovere noise a blocchi
    disparity_map = cv2.medianBlur(disparity_map, 5)

    #disparity_map[disparity_map < 1.0] = 0           
    return disparity_map

def compute_depth_map(disparity, k_left, baseline, Q):
    
    #imposto soglia profondità massima: se la supero, valore di profondità non valido, viene posto pari a 0
    max_depth=1000

    #clipping valori errati, in particolare si vogliono rimuovere:
    #valori troppo bassi, che porterebbero a profondità eccessivamente elevate
    #valore nullo, indicatore di match non trovato
    #valori negativi, indicatori di mismatch
    #il valore 0 indica ora, quindi, dato da non considerare dagli algoritmi successivi
    disparity[disparity<=0]= 0.1
    
    #depth_map= np.zeros(disparity.shape)
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    depth_map = points_3D[:, :, 2]  # Z è la profondità
    depth_map[depth_map > max_depth] = 0

    
    return depth_map


def decomposeProjectionMatrix(P):
    k, r, t, _, _, _,_= cv2.decomposeProjectionMatrix(P)
    t=(t/t[3])[:3]
    return k,r,t

def stereo_to_depth(img_left,img_right,P0,P1,baseline,Q,matcher="sgbm"):
    #compute the disparity map
    disparity_map= compute_disparity_map(img_left,img_right,matcher_name=matcher)

    #compute k,r,t
    k_left,r_left,t_left= decomposeProjectionMatrix(P0)

    #compute the depth map
    depth_map= compute_depth_map(disparity_map, k_left, baseline,Q)
    
    return depth_map


