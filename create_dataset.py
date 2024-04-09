import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import csv
from cv2 import VideoCapture

from utils import draw_landmarks_on_image


MODEL_PATH = '/home/tomek/VisualStudio/PaintOnline/InteractivePaint/ImageProcessing/MediaPipe/models/hand_landmarker.task'      #TODO do env

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

fields = ['x', 'y', 'z', 'visibility', 'presence']
path = 'dataset/fist/'
filename = 'fist.csv'  

  
cam = VideoCapture(0)

num = 0
max_num = 500
dataset = []
while num < max_num:
    result, image = cam.read()
    if result:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        detection_result = detector.detect(mp_image)
        if detection_result.hand_landmarks and len(detection_result.hand_landmarks[0]) == 21:
            points = {f'{field}_{idx}': getattr(landmark, field) for idx, landmark in enumerate(detection_result.hand_landmarks[0]) for field in fields}
            points['id'] = num
            image_with_landmarks = draw_landmarks_on_image(image, detection_result)
            cv2.imshow('xxd', image_with_landmarks)
            key = cv2.waitKey(0)
            if key == 32:
                cv2.imwrite(f'{path}{num}.jpg', image_with_landmarks)
                dataset.append(points)
                num += 1
                print(f"Image number {num} saved!")

fields = list(dataset[0].keys())

with open(path+filename, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=fields)

    writer.writeheader()
    writer.writerows(dataset)
