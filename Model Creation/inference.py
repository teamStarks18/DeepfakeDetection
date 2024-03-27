import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
from torch import nn
import os
import numpy as np
from torchvision import models
import mediapipe as mp

# Constants
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax()
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))


# Model definition
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        self.model = nn.Sequential(*list(models.resnext50_32x4d(pretrained=True).children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


# Function to process video using Mediapipe for face detection
def process_video(video_path, sequence_length=20, transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])):
    frames = []
    a = int(100 / sequence_length)
    first_frame = np.random.randint(0, a)
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()

    for i, frame in enumerate(frame_extract(video_path)):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_detection.process(frame_rgb)
        if result.detections:
            for detection in result.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                frame = frame[y:y + h, x:x + w]
        frames.append(transform(frame))
        if len(frames) == sequence_length:
            break
    frames = torch.stack(frames)
    frames = frames[:sequence_length]
    return frames.unsqueeze(0)


# Function to make predictions
def predict(model, video_object):
    fmap, logits = model(video_object.to('cuda'))
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return fmap, logits, weight_softmax, confidence


# Function to extract frames from video
def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image


def process_and_predict(video_path):
    video_object = process_video(video_path, sequence_length=20)
    model = Model(2).cuda()
    path_to_model = "detectors/weights/model1.pt"
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    fmap, logits, weight_softmax, con = predict(model, video_object)
    return con, "model1"

# print(process_and_predict(r"path"))
