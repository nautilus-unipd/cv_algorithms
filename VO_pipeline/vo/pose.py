import numpy as np
import cv2



def estimate_motion(kp1,kp2,k,matches,depth, max_depth=3000):
    #compute the image in pixel  
    img_points1= [kp1[m.queryIdx].pt for m in matches]
    img_points2= [kp2[m.trainIdx].pt for m in matches]

    # extracting parameters from k
    fx=k[0][0]
    fy=k[1][1]
    cx=k[0][2]
    cy=k[1][2]

    #converting interesting image_points1 from 2d to 3d
    objects_list=[]
    valid_img_points1=[]
    valid_img_points2=[]

    for i , (u,v) in enumerate(img_points1):
        z= depth[int(round(v))][int(round(u))]

        #check su valore max (ridondante)
        if z >= max_depth:
            continue

        x= z*(u-cx)/fx
        y= z*(v-cy)/fy
        objects_list.append([x,y,z])
        valid_img_points1.append(img_points1[i])
        valid_img_points2.append(img_points2[i])

    if(len(objects_list) < 4):
        print("Non abbastanza punti 3D\n")
        return None, None

    objects_list=np.array(objects_list, dtype=np.float32)
    valid_img_points1=np.array(valid_img_points1, dtype=np.float32)
    valid_img_points2=np.array(valid_img_points2, dtype=np.float32)
    
    ret, rvec,tvec, _= cv2.solvePnPRansac(objects_list, valid_img_points2, k, None)

    if not ret:
        
        print("Errore in estimate motion\n")
        return None, None
        
    rvec = cv2.Rodrigues(rvec)[0]
    
    return rvec, tvec
