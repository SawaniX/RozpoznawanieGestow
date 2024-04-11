import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import cv2
import csv
import time
import numpy as np
import os
from cv2 import VideoCapture

from utils import draw_landmarks_on_image, coordinates_to_image, ImageShape


MODEL_PATH = '/home/tomek/VisualStudio/PaintOnline/InteractivePaint/ImageProcessing/MediaPipe/models/hand_landmarker.task'      #TODO do env

fields = ['x', 'y', 'z', 'visibility', 'presence']
path = 'dataset/moj/call'
filename = 'fist.csv'  

class DatasetCreator:
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                                num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    def __init__(self, samples_num: int) -> None:
        self.cam = VideoCapture(0)
        self.samples_num = samples_num

    def create_csv_dataset(self) -> None:
        dataset = []
        saved_num = 0
        while saved_num < self.samples_num:
            result, image = self.cam.read()
            if not result:
                continue
            detection_result = self._detect_landmarks(image)
            if not detection_result:
                continue
            points = {f'{field}_{idx}': getattr(landmark, field) for idx, landmark in enumerate(detection_result.hand_landmarks[0]) for field in fields}
            points['id'] = saved_num
            image_with_landmarks = draw_landmarks_on_image(image, detection_result)
            cv2.imshow('Sample', image_with_landmarks)
            key = cv2.waitKey(0)
            if key == 32:
                cv2.imwrite(f'{path}{saved_num}.jpg', image_with_landmarks)
                dataset.append(points)
                saved_num += 1
                print(f"Image number {saved_num} saved!")

    def create_images_dataset(self, output_shape: ImageShape, sleep_time: float) -> None:
        saved_num = 0
        while saved_num < self.samples_num:
            result, image = self.cam.read()
            if not result:
                continue
            detection_result = self._detect_landmarks(image)
            if not detection_result:
                continue
            landmarks = detection_result.hand_landmarks[0]
            x_norm, y_norm = self._normalize_landmarks(landmarks)
            sample = coordinates_to_image(output_shape, x_norm, y_norm)
            cv2.imwrite(os.path.join(path, f'{saved_num}.jpg'), sample)
            cv2.imshow('Sample', sample)
            cv2.waitKey(sleep_time)
            saved_num += 1
            
    def _normalize_landmarks(self, landmarks: list[NormalizedLandmark]) -> list[int]:
        x = [l.x for l in landmarks]
        y = [l.y for l in landmarks]
        min_x, min_y, max_x, max_y = min(x), min(y), max(x), max(y)
        x_norm = [int((value - min_x) / (max_x - min_x) * (output_shape.x - 1)) for value in x]
        y_norm = [int((value - min_y) / (max_y - min_y) * (output_shape.y - 1)) for value in y]
        return x_norm, y_norm

    def _detect_landmarks(self, image: np.ndarray) -> list|HandLandmarkerResult:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(mp_image)
        if not detection_result.hand_landmarks or len(detection_result.hand_landmarks[0]) != 21:
            return []
        return detection_result

    def _write_to_csv(self, path: str, dataset: list) -> None:
        with open(path+filename, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(dataset)
            
            
if __name__=='__main__':
    samples_num = 500
    sleep_time = 100
    output_shape = ImageShape(28, 28)
    
    DatasetCreator(samples_num).create_images_dataset(output_shape, sleep_time)
