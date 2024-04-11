import pandas as pd
import numpy as np
import cv2
import json
import os
from abc import ABC, abstractmethod

from dataset_prepare.utils import LANDMARKS_NUM, ImageShape, coordinates_to_image


GESTURES: list[str] = [
    'call',
    'dislike',
    'fist',
    'four',
    'like',
    'ok',
    'one',
    'palm',
    'peace',
    'rock',
    'stop',
    'three',
    'three2',
    'two_up'
]


class Processor(ABC):
    def __init__(self, output_shape: ImageShape, dataset_path: str) -> None:
        self.output_shape = output_shape
        self.dataset_path = dataset_path

    @abstractmethod
    def process(self) -> None:
        pass

    @abstractmethod
    def _normalize_landmarks(self, landmarks: list[list[float]]):
        pass

    @abstractmethod
    def _load_gesture_dataset(self, path: str) -> dict:
        pass


class OwnDatasetProcessor(Processor):
    def __init__(self, output_shape: ImageShape, dataset_path: str) -> None:
        super().__init__(output_shape, dataset_path)

    def process(self) -> None:
        for gesture in GESTURES:
            path = self.dataset_path + gesture + '/'
            dataset = self._load_gesture_dataset(path)

            for _, data in dataset.iterrows():
                x_norm, y_norm = self._normalize_landmarks(data)

                sample = coordinates_to_image(self.output_shape, x_norm, y_norm)
                id = int(data['id'])
                cv2.imwrite(os.path.join(path, f'{id}.jpg'), sample)
    
    def _normalize_landmarks(self, landmarks):
        x = [landmarks[f'x_{i}'] for i in range(LANDMARKS_NUM)]
        y = [landmarks[f'y_{i}'] for i in range(LANDMARKS_NUM)]
        min_x, min_y, max_x, max_y = min(x), min(y), max(x), max(y)
        x_norm = [int((value - min_x) / (max_x - min_x) * (self.output_shape.x - 1)) for value in x]
        y_norm = [int((value - min_y) / (max_y - min_y) * (self.output_shape.y - 1)) for value in y]
        return x_norm, y_norm
    
    def _load_gesture_dataset(self, path: str, gesture: str) -> dict:
        return pd.read_csv(path + f'{gesture}.csv')


class DownloadedDatasetProcessor(Processor):
    def __init__(self, output_shape: ImageShape, dataset_path: str) -> None:
        super().__init__(output_shape, dataset_path)

    def process(self) -> None:
        for gesture in GESTURES:
            path = os.path.join(self.dataset_path, gesture)
            dataset = self._load_gesture_dataset(path, gesture)
            print(len(dataset))
            for sample_id, data in dataset.items():
                labels = data.get('labels')
                if len(labels) != 1 or gesture not in labels:
                    continue
                landmarks = data.get('landmarks')[0]
                if len(landmarks) != LANDMARKS_NUM:
                    continue
                x_norm, y_norm = self._normalize_landmarks(landmarks)
                sample = coordinates_to_image(self.output_shape, x_norm, y_norm)
                cv2.imwrite(os.path.join(path, f'{sample_id}.jpg'), sample)
    
    def _normalize_landmarks(self, landmarks: list[list[float]]):
        x = [l[0] for l in landmarks]
        y = [l[1] for l in landmarks]
        min_x, min_y, max_x, max_y = min(x), min(y), max(x), max(y)
        x_norm = [int((value - min_x) / (max_x - min_x) * (self.output_shape.x - 1)) for value in x]
        y_norm = [int((value - min_y) / (max_y - min_y) * (self.output_shape.y - 1)) for value in y]
        return x_norm, y_norm

    def _load_gesture_dataset(self, path: str, gesture: str) -> dict:
        dataset = open(os.path.join(path, f'{gesture}.json'))
        return json.load(dataset)
    

if __name__=='__main__':
    output_shape = ImageShape(28, 28)
    DOWNLOADED_PATH = 'dataset/gotowy'
    OWN_PATH = 'dataset/moj'

    #DownloadedDatasetProcessor(output_shape, DOWNLOADED_PATH).process()
    OwnDatasetProcessor(output_shape, OWN_PATH).process()
