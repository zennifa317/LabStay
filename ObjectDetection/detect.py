import argparse

import cv2
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection with SSDLite320 MobileNet V3")
    parser.add_argument('--source', type=str, default='data/images', help='Path to the source images or video')
    args = parser.parse_args()
    
    weight = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weight)
    model.eval()