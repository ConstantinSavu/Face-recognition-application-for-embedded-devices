import mediapipe as mp
from mediapipe import Image

import cv2
import numpy as np
import os



class Features:
    def __init__(self, name, image_path, model) -> None:
        self.name = name
        self.image_path = image_path
        self.feature = self.get_features(image_path, model)
    
    def get_features(self, image_path, model):
    
        cv_mat = cv2.imread(image_path)
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)
        return model.embed(rgb_frame).embeddings[0].embedding




def get_folder_features(name, path, model):
    files = (file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file)))
    
    files = list(files)
    files.sort()
    folder_features = []
    for file in files:
        folder_features.append(
            Features(
                name,
                os.path.join(path, file),
                model
            )
        )
    
    return folder_features

def get_all_features(path, model):
    dirs = (file for file in os.listdir(path) if os.path.isdir(os.path.join(path, file)))

    dirs = list(dirs)
    dirs.sort()

    all_features = []

    for dir in dirs:
        all_features.extend(
            
            get_folder_features(dir, os.path.join(path, dir, "faces"), model)

        )
    
    return all_features



