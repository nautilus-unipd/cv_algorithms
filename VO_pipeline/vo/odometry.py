import numpy as np
import cv2
from . import Depth
from . import tracking
from . import pose

def visual_odometry(handler, stereo_matcher="bm", detector= "orb", matcher="BF", filter_threshold=None):
              
    
    k_left,r_left,t_left = Depth.decomposeProjectionMatrix(handler.P0)

    trajectory=np.zeros((handler.num_frames, 3,4))
    T_tot=np.eye(4)
    trajectory[0]=(T_tot[:3,:])

    #Compute the trajectory for every frame
    for i in range(handler.num_frames-1):
        
        left_img=handler.rec_left[i]
        right_img=handler.rec_right[i]
        left_imgplus1=handler.rec_left[i+1]

        depth= Depth.stereo_to_depth(left_img, right_img, handler.P0, handler.P1, handler.baseline, handler.Q)

        kp0,des0 = tracking.extract_features(left_img)
        kp1,des1 = tracking.extract_features(left_imgplus1)
        
        unmatched_features= tracking.match_features(des0,des1)

        if filter_threshold is not None :
            matched_features=tracking.filter_matches(unmatched_features, filter_threshold)
        else:
            matched_features= unmatched_features #problem : list of 2-element list, ransac doesn't want it
        
        rvec,tvec= pose.estimate_motion(kp0,kp1,k_left, matched_features, depth)

        if rvec is None or tvec is None:
            print(f"Skipping frame {i+1} due to failed motion estimation.")
            trajectory[i+1] = trajectory[i]
            continue
        
        #computing the transformation matrix [rvec| tvec]
        T_mat= np.hstack([rvec,tvec])

        #computing the homogeneous transformation matrix, so then I can invert it
        T_mat=np.vstack([T_mat, [0,0,0,1]])

        #inverting the transformation matrix
        T_mat=np.linalg.inv(T_mat)

        T_tot = T_tot.dot(T_mat)

        #appending T_tot into trajectory

        trajectory[i+1, :, :] = T_tot[:3][:]
        print(f"Frame {i+1} processato.")
    
    print(trajectory)

    return trajectory