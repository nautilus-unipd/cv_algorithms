import numpy as np
from tqdm import tqdm
import cv2
import os
from matplotlib import pyplot as plt


class Dataset_Handler:
    def __init__(self,calib_path, image_path):
        self.image_size_calib = (1920, 1080)
        self.image_size_runtime = (1242, 376)
        self.getCalibParams(calib_path)
        self.getImages(image_path)
        #self.getPoses(gtPoses_path)
    
    def getCalibParams(self,filepath):
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
        #self.P0 = fs.getNode("P1").mat()
        #self.P1 = fs.getNode("P2").mat()
        self.cameraMatrix1= fs.getNode("cameraMatrix1").mat()
        self.distCoeffs1= fs.getNode("distCoeffs1").mat()
        self.cameraMatrix2= fs.getNode("cameraMatrix2").mat()
        self.distCoeffs2= fs.getNode("distCoeffs2").mat()
        self.R=fs.getNode("R").mat()
        self.T=fs.getNode("T").mat()
        self.baseline=np.linalg.norm(np.array(self.T, dtype=np.float32))
        fs.release()

        self.cameraMatrix1=np.array(self.cameraMatrix1, dtype=np.float64)
        self.cameraMatrix2=np.array(self.cameraMatrix2, dtype=np.float64)
        self.distCoeffs1=np.array(self.distCoeffs1, dtype=np.float64)
        self.distCoeffs2=np.array(self.distCoeffs2, dtype=np.float64)
        self.R=np.array(self.R, dtype=np.float64)
        self.T=np.array(self.T, dtype=np.float64)

        scale_x= self.image_size_runtime[0] / self.image_size_calib[0]
        scale_y= self.image_size_runtime[1] / self.image_size_calib[1]
        self.cameraMatrix1[0,0]*=scale_x
        self.cameraMatrix1[1,1]*=scale_y
        self.cameraMatrix1[0,2]*=scale_x
        self.cameraMatrix1[1,2]*=scale_y

        self.cameraMatrix2[0,0]*=scale_x
        self.cameraMatrix2[1,1]*=scale_y
        self.cameraMatrix2[0,2]*=scale_x
        self.cameraMatrix2[1,2]*=scale_y
        

    def getImages(self, filepath):
       
        #caricamento delle immagini
        self.left_images = np.array(self.loadImages(filepath+"/left/"), dtype=np.uint8)
        self.right_images = np.array(self.loadImages(filepath+"/right/"), dtype=np.uint8)

        #check per verificare che tutte le immagini abbiano la risoluzione corretta
        for img in self.left_images + self.right_images:
            assert img.shape == self.image_size_runtime[::-1], "Problems with images shape"
            assert img.dtype == np.uint8, "Problems with images dtype value"
            
        self.num_frames= len(self.left_images)
        self.RectifyImages()

    def loadImages(self,directory):
        image_names = sorted([f for f in os.listdir(directory)])
        return [cv2.imread(os.path.join(directory,img),cv2.IMREAD_GRAYSCALE) for img in tqdm(image_names, desc=f"Loading images from {directory}")]
             

    def RectifyImages(self):
        #calcolo dei parametri di rettifica
        R1, R2, self.P0, self.P1, self.Q, roi1, roi2 = cv2.stereoRectify(
        self.cameraMatrix1, self.distCoeffs1,
        self.cameraMatrix2, self.distCoeffs2,
        self.image_size_runtime,  # (width, height)
        self.R, self.T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
        )
        # Calcola le mappe di rettifica
        map1_x, map1_y = cv2.initUndistortRectifyMap(
            self.cameraMatrix1, self.distCoeffs1, R1, self.P0, self.image_size_runtime, cv2.CV_16SC2)
        map2_x, map2_y = cv2.initUndistortRectifyMap(
            self.cameraMatrix2, self.distCoeffs2, R2, self.P1, self.image_size_runtime, cv2.CV_16SC2)
        
        # Applica la mappatura per rettificare le immagini
        self.rec_left=[]
        self.rec_right=[]
        for i in tqdm(range(self.num_frames), desc="Rectificating images..."):
            rectified_left = cv2.remap(self.left_images[i], map1_x, map1_y, interpolation=cv2.INTER_LINEAR)
            rectified_right = cv2.remap(self.right_images[i], map2_x, map2_y, interpolation=cv2.INTER_LINEAR)
            self.rec_left.append(rectified_left)
            self.rec_right.append(rectified_right)

    
    def getPoses(self, filepath):
        df=pd.read_csv(filepath, delimiter=" ", header=None)
        self.poses=[]
        for i in range(df.shape[0]):
            arr=np.ascontiguousarray(df.loc[i].to_numpy())
            arr.resize(3,4)
            self.poses.append(arr)
        self.poses= np.array(self.poses)