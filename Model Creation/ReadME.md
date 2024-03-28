<!-- Header Section -->
# Deepfake Detection Model Creation


<!-- Dataset Section -->
## Dataset
The following datasets were utilized in training our model:
- [FaceForensics](https://github.com/ondyari/FaceForensics)
- [CelebDF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/data)


<!-- Dataset Icon -->
<img src="https://github.com/teamStarks18/DeepfakeDetection/blob/main/images/dataseticon.png" alt="Dataset Icon" width="150" height="150">

## Custom Dataset
Randomly selected an equal number of videos from all the mentioned datasets. Various augmentation techniques were implemented to increase the number of samples in the dataset.

## Preprocessing
All the programs used for data preprocessing and augmentation are located under the [Preprocessing Dataset](https://github.com/teamStarks18/DeepfakeDetection/tree/main/Model%20Creation/preprocessing_dataset) directory. The approach involved extracting all frames from the video, using Mediapipe to detect faces in each frame, adding required padding to the region of interest to gather more information from around the face and the face itself while ignoring the rest of the video. These cropped images are then combined to form the preprocessed video. Techniques such as horizontal flipping were used to increase the number of inputs.

<!-- Loading Data Section -->
## Loading the Data
All the videos are converted into 5D vectors through the following steps:
1. **PyTorch Utilities:** PyTorch utilities such as Dataset Class, DataLoader, and Transforms were employed to create the data pipeline, loading the data into memory in batches.
2. **Transforms:** The transforms include filters like resizing, normalizing, and then converting into tensors.
3. **Data Loader Output:** The DataLoader returns a 5D output on each iteration where the dimensions are batch size, sequence length, channel value, height, and width. Therefore, the output from the DataLoader is unpacked and fed into the model sequentially.

<!-- Model Architecture Section -->
## Model Architecture
We have experimented with various architectures such as ResNet-101, VGG16, DenseNet121, ResNet-152, DenseNet-201, and VGG19, along with their different depth variants, to observe the effect of increasing depth and architecture design in deepfake detection. Multiple models were created and trained, each employing a different architecture or dataset.

<!-- Pretrained Models and Code Links -->
The trained models are available in the [Pretrained Models](https://drive.google.com/drive/folders/1o4lNbL9odOtQoXiELppH3z4IUuAV30fn?usp=sharing) directory. The code for training and testing can be found [here](https://github.com/teamStarks18/DeepfakeDetection/blob/main/Model%20Creation/train.ipynb). Additionally, the code for inference is available [here](https://github.com/teamStarks18/DeepfakeDetection/blob/main/Model%20Creation/inference.py).

<!-- Training Section -->
## Training
A batch size of 4 is utilized, with each input having a sequence length of 60, and 20 epochs are allocated for training all the developed models.

<!-- Evaluation Section -->
## Evaluation
We have opted to use confusion matrices, training-validation curves, and accuracy as factors to evaluate the models.
