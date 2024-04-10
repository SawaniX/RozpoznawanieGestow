import pandas as pd
import numpy as np
import cv2
from dataclasses import dataclass
import json
import os
from abc import ABC, abstractmethod


@dataclass
class ImageShape:
    x: int
    y: int

    def __iter__(self):
        yield self.x
        yield self.y

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
LANDMARKS_NUM = 21
LANDMARKS_LINKS = {
    0: [1, 5, 17],
    1: [2],
    2: [3],
    3: [4],
    5: [6, 9],
    6: [7],
    7: [8],
    9: [10, 13],
    10: [11],
    11: [12],
    13: [14, 17],
    14: [15],
    15: [16],
    17: [18],
    18: [19],
    19: [20]
}


class Processor(ABC):
    def __init__(self, output_shape: ImageShape, dataset_path: str) -> None:
        self.output_shape = output_shape
        self.dataset_path = dataset_path

    @abstractmethod
    def process(self) -> None:
        pass

    def _create_sample(self, x_norm: list[int], y_norm: list[int]) -> np.ndarray:
        blank_img = np.zeros(tuple(self.output_shape), np.uint8)
        for idx_from, target in LANDMARKS_LINKS.items():
            for idx_to in target:
                blank_img = cv2.line(blank_img, (x_norm[idx_from], y_norm[idx_from]), (x_norm[idx_to], y_norm[idx_to]), (255, 255, 255), 1)
        return blank_img

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

                sample = self._create_sample(x_norm, y_norm)
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
                sample = self._create_sample(x_norm, y_norm)
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

    DownloadedDatasetProcessor(output_shape, DOWNLOADED_PATH).process()
