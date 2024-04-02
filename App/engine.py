import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import threading
import os
from preprocessor import face_extractor

from detectors import (
    denseNet121,
    efficientnet_b0,
    efficientnet_b0_pixel_200_seq_10,
    efficientnet_b3,
    resnet101,
    resnext50_32x4d,
    resnext50_32x4d_pixel_200_seq_5,
    vgg16,
    vgg19,
    vgg19_pixel_200_seq_5,
)


class Aggregate(nn.Module):
    def __init__(self, input_size):
        super(Aggregate, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.7)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.7)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))
        return x


class Detector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.results_from_models = {}
        self.detector_list = [
            denseNet121,
            efficientnet_b0,
            efficientnet_b0_pixel_200_seq_10,
            efficientnet_b3,
            resnet101,
            resnext50_32x4d,
            resnext50_32x4d_pixel_200_seq_5,
            vgg16,
            vgg19,
            vgg19_pixel_200_seq_5,
        ]
        self.preprocessed_video = face_extractor.extract_frames_and_combine_to_video(
            self.video_path, r"output_videos_real", 100, 1.5, (256, 256)
        )

    def predict_model(self, model):
        model_name = model.__name__
        self.results_from_models[model_name] = model.predict(self.preprocessed_video)

    def individual_predictions(self):
        threads = []
        for model in self.detector_list:
            t = threading.Thread(target=self.predict_model, args=(model,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return {
            key.replace("detectors.", ""): value
            for key, value in self.results_from_models.items()
        }

    def aggregate(self):
        predictions = self.individual_predictions()
        inputs = torch.tensor(
            np.array([predictions[model_name] for model_name in predictions])
        )
        model = Aggregate(input_size=len(predictions))
        model.load_state_dict(torch.load("aggregate.pth"))
        model.eval()

        # Perform inference
        with torch.no_grad():
            output = model(inputs)

        predictions["aggregate"] = output.item()  # Change output.value to output.item()
        return predictions


# # Define the path to the video file
# video_path = (
#     r"C:\Users\mahes\Downloads\WhatsApp Video 2024-04-01 at 23.41.16_66d08784.mp4"
# )

# b = time.time()
# detector = Detector(video_path)

# print(detector.aggregate())

# print(time.time() - b)

# for i in os.listdir(r"C:\Users\mahes\ML\new Deepfake Train Custom\testsample"):
#     detector = Detector(rf"C:\Users\mahes\ML\new Deepfake Train Custom\testsample\{i}")
#     print(rf"C:\Users\mahes\ML\new Deepfake Train Custom\testsample\{i}")
#     print(detector.aggregate())
#     print()
