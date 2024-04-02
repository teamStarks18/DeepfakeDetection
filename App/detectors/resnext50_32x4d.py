import torch
from torch import nn
import cv2
from torchvision import transforms
from torchvision.models import resnext50_32x4d


class Model(nn.Module):
    def __init__(
        self,
        num_classes,
        latent_dim=2048,
        lstm_layers=1,
        hidden_dim=2048,
        bidirectional=False,
    ):
        super(Model, self).__init__()
        # Load pre-trained ResNeXt-50 model
        model = resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
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
        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))


# Define the transformations for inference
im_size = 150
mean = [0.30744792, 0.33174657, 0.44394907]
std = [0.17503524, 0.18142716, 0.23555627]
inference_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


# Function to perform inference on a single video
def inference(video_path, model, sequence_length=30, transform=inference_transforms):
    frames = []
    vidObj = cv2.VideoCapture(video_path)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            frames.append(transform(image))
            if len(frames) == sequence_length:
                break
    frames = torch.stack(frames)
    frames = frames[:sequence_length]
    # Add batch dimension
    frames = frames.unsqueeze(0)
    # Perform inference
    with torch.no_grad():
        model.eval()
        _, outputs = model(frames)
        probabilities = torch.softmax(outputs, dim=1)
        return probabilities.squeeze().cpu().numpy()


def predict(video_path):
    model = Model(num_classes=2)
    model.load_state_dict(torch.load(r"detectors\resnext50_32x4d.pt"))
    model.eval()
    probabilities = inference(video_path, model)
    fake_probability = probabilities[0]
    # real_probability = probabilities[1]
    return fake_probability


# # Example usage:
# video_path1 = (
#     r"C:\Users\mahes\ML\new Deepfake Train Custom\sample fake\fake_video67.mp4"
# )
# video_path2 = (
#     r"C:\Users\mahes\ML\new Deepfake Train Custom\sample fake\fake_video72.mp4"
# )
# video_path3 = (
#     r"C:\Users\mahes\ML\new Deepfake Train Custom\sample real\real_video69.mp4"
# )
# video_path4 = (
#     r"C:\Users\mahes\ML\new Deepfake Train Custom\sample real\real_video73.mp4"
# )

# fake_prob1, real_prob1 = predict_resnext50(video_path1)
# print("Fake Probability (Video 1):", fake_prob1)
# print("Real Probability (Video 1):", real_prob1)

# fake_prob2, real_prob2 = predict_resnext50(video_path2)
# print("Fake Probability (Video 2):", fake_prob2)
# print("Real Probability (Video 2):", real_prob2)

# fake_prob3, real_prob3 = predict_resnext50(video_path3)
# print("Fake Probability (Video 3):", fake_prob3)
# print("Real Probability (Video 3):", real_prob3)

# fake_prob4, real_prob4 = predict_resnext50(video_path4)
# print("Fake Probability (Video 4):", fake_prob4)
# print("Real Probability (Video 4):", real_prob4)
