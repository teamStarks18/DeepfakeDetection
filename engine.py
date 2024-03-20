import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import os
import glob
import importlib.util

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Ensure that the bias type matches the input type
        x = self.fc2(x.type_as(self.fc2.bias))
        x = self.sigmoid(x)
        return x

class Detector:
    def __init__(self, video_path, detectors_folder, aggregate_model_weights):
        self.video_path = video_path
        self.detectors_folder = detectors_folder
        self.aggregate_model_weights = aggregate_model_weights
        self.results_from_models = {}
        self.no_detectors = len(glob.glob(os.path.join(self.detectors_folder, "model*.py")))
        self.detector_list = [f"model{i}" for i in range(1 , self.no_detectors+1)]

    def load_models(self):
        models = {}
        model_files = glob.glob(os.path.join(self.detectors_folder, "model*.py"))
        for model_file in model_files:
            model_name = os.path.splitext(os.path.basename(model_file))[0]
            spec = importlib.util.spec_from_file_location(model_name, model_file)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            models[model_name] = model_module.process_and_predict
        return models

    def model_inference(self, model_name, predict_function):
        result, name = predict_function(self.video_path)
        self.results_from_models[name] = result

    def individual_probabilities(self, models):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.model_inference, model_name, predict_function) 
                       for model_name, predict_function in models.items()]
            for future in futures:
                future.result()

    def aggregate(self):
        models = self.load_models()
        self.individual_probabilities(models)
        self.results_from_models = dict(sorted(self.results_from_models.items()))

        # Create an instance of the model
        model = CNNModel()
        # Load the saved model weights
        model.load_state_dict(torch.load(self.aggregate_model_weights))
        model.eval()

        # Define the mean and standard deviation
        mean = torch.tensor([[[95.93638682]]])
        std = torch.tensor([[[9.84827027]]])

        result = [self.results_from_models[detector_name] for detector_name in self.detector_list]

        # Input data (replace this with your actual input)
        new_input = torch.tensor(result)  # Adjust the shape according to your model input

        # Normalize the input using the provided mean and standard deviation
        normalized_input = (new_input - mean) / std

        # Perform inference
        with torch.no_grad():
            output_probability = model(normalized_input)

        aggregate_output = float(output_probability)
        self.results_from_models["aggregated_probability"] = output_probability.item() * 100
        return self.results_from_models
    
   
# Path to the models folder
detectors_folder = "detectors/models"
aggregate_model_weights = "cnn_model.pth"
video_path = "sample_inputs/fake_inputs/id0_id1_0000.mp4"

d = Detector(video_path, detectors_folder, aggregate_model_weights)
print(d.aggregate())
