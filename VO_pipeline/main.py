from data import loader
from vo import odometry
import numpy as np

def main():
    calib_path = "data/calib_images/CalibrationParameters3.yaml"
    image_path="data/DatasetProva"
    handler = loader.Dataset_Handler(calib_path, image_path)
    assert handler.left_images[0].dtype == np.uint8
    trajectory= odometry.visual_odometry(handler, filter_threshold=0.5)

if __name__ == "__main__":
    main()