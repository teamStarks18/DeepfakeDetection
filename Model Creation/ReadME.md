<!-- Header Section -->
# Deepfake Detection Model Creation


<!-- Dataset Section -->
## Dataset
The open source dataset used in the project can be downloaded from here:
- [FaceForensics](https://github.com/ondyari/FaceForensics)
- [CelebDF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/data)


## Custom Dataset
We opted for CelebDF and FF++ datasets to construct our custom dataset due to their balanced nature, larger quantity of videos, and higher quality deepfakes.

A 50-50 split was executed on CelebDF and FF++ datasets, resulting in a collection of 3,780 videos for each label (Real and Fake), totaling 7,560 videos. An 80-20 train-test split was then applied to this dataset. During training, a batch size of 4 and 20 epochs were utilized for the detectors.

- Fake
    - CelebDF -  1890 
    - FF++ - 
        - Deepfakes - 630 
        - Face 2 Face - 630
        - FaceSwap - 630
For the "Fake" category, 1,890 videos were sourced from CelebDF, while from FF++, 630 videos each were taken from Deepfakes, Face2Face, and FaceSwap categories. Regarding the "Real" category, all genuine videos from both datasets were used, excluding those featuring multiple individuals. The number of fake videos was adjusted to match the available real videos.

Additionally, augmentation techniques like horizontal flipping were applied to the real videos to expand the available video pool.


## Preprocessing
All the programs used for data preprocessing and augmentation are located under the [Preprocessing Dataset](https://github.com/teamStarks18/DeepfakeDetection/tree/main/Model%20Creation/preprocessing_dataset) directory. The approach involved extracting all frames from the video, using Mediapipe to detect faces in each frame, adding required padding to the region of interest to gather more information from around the face and the face itself while ignoring the rest of the video. These cropped images are then combined to form the preprocessed video. This helps to grealty reduce the size of the dataset.

## Sample of Preprocessed Data

![video20-ezgif com-video-to-gif-converter](https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/d1028181-8d03-4046-9168-6ad7d6435731)
![video50-ezgif com-video-to-gif-converter](https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/59225353-d4f3-4dc4-a777-1e490497525e)
![video250-ezgif com-video-to-gif-converter](https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/a440b2c3-65bd-49d6-956f-744ff712f522)
![video45-ezgif com-video-to-gif-converter](https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/830f6a32-ebb3-4ce5-94b1-1dcbaefa81ce)
![video1157-ezgif com-video-to-gif-converter](https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/4652e3fd-aed9-40d8-8f72-e1a77c75788a)
![video182-ezgif com-video-to-gif-converter](https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/b9750687-eee4-450f-9baf-0b69bee00133)







<!-- Loading Data Section -->
## Loading the Data
All the videos are converted into 5D vectors through the following steps:
1. **PyTorch Utilities:** PyTorch utilities such as Dataset Class, DataLoader, and Transforms were employed to create the data pipeline, loading the data into memory in batches.
2. **Transforms:** The transforms include filters like resizing, normalizing, and then converting into tensors.
3. **Data Loader Output:** The DataLoader returns a 5D output on each iteration where the dimensions are batch size, sequence length, channel value, height, and width. Therefore, the output from the DataLoader is unpacked and fed into the model sequentially.

<!-- Model Architecture Section -->
## Model Architecture
We utilized various pretrained architectures for feature extraction in each detector, including:

- DenseNet121
- EfficientNet_B0
- EfficientNet_B3
- ResNet101
- ResNext50_32x4d
- VGG16
- VGG19. 
Each architecture incorporates an LSTM layer to capture sequential data. These models were trained on different input dimensions and sequence lengths, taking into account computational resources and accuracy requirements.

Utilizing a variety of pretrained architectures in a single detector offers significant advantages. Firstly, it enables diversity in feature representation. Each architecture is adept at capturing distinct aspects of the input data, thereby generating a diverse set of features. 

Secondly, employing multiple architectures enhances the detector's robustness to model biases. Different architectures may exhibit biases towards specific types of features or patterns. By leveraging a range of architectures, the detector becomes more resilient to these biases. This heightened resilience enhances the detector's ability to generalize effectively across diverse data distributions, thereby bolstering its overall performance and reliability.

While employing various architectures is beneficial, training on a diverse dataset with high-quality deepfakes remains pivotal for enhancing the detector's generalizability.


<!-- Pretrained Models and Code Links -->
The trained models are available in the [Pretrained Models](https://drive.google.com/drive/folders/1o4lNbL9odOtQoXiELppH3z4IUuAV30fn?usp=sharing) directory. The code for training and testing can be found [here](https://github.com/teamStarks18/DeepfakeDetection/blob/main/Model%20Creation/train.ipynb). Additionally, the code for inference is available [here](https://github.com/teamStarks18/DeepfakeDetection/blob/main/Model%20Creation/inference.py).

<!-- Training Section -->
## Training
A batch size of 4 is utilized, with each input having varying sequence length for each detector and 20 epochs are used for training all the developed models.

<!-- Evaluation Section -->
## Evaluation
We have opted to use confusion matrices, training-validation curves, and accuracy as factors to evaluate the models.

The accuracies obtained were:

| Model            | Accuracy |
|------------------|----------|
| DenseNet121      | 91       |
| EfficientNet_B0  | 89       |
| EfficientNet_B3  | 81       |
| ResNet101        | 88       |
| ResNext50_32x4d  | 92       |
| VGG16            | 86       |
| VGG19            | 87       |





## Training Validation Curves And Confusion Matrix

## Training Validation Curves And Confusion Matrix

### DenseNet121:
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/e60b588f-c3bd-46b5-89e7-02e1f8cc89fa" alt="Accuracy" style="width: 30%; margin: 0 5px;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/51b2596f-5a17-4f26-a759-1abc371ec9c6" alt="Loss" style="width: 30%; margin: 0 5px;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/eedad236-18ad-43e5-b7f9-adb55c26add2" alt="Confusion Matrix" style="width: 30%; margin: 0 5px;">
</div>

### EfficientNet_B0:
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/96e24ec0-7706-4aea-bdce-c3e209b0dddb" alt="Accuracy" style="width: 30%; margin: 0 5px;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/03be4ad9-0f9d-442b-8a50-316accac3ef2" alt="Loss" style="width: 30%; margin: 0 5px;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/77687880-0688-4ea0-9674-005d17f7ee1a" alt="Confusion Matrix" style="width: 30%; margin: 0 5px;">
</div>

### EfficientNet_B3:
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/9ed48b0d-6c8e-4690-886a-f68d910039dc" alt="Accuracy" style="width: 30%; margin: 0 5px;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/c3d639bd-79d8-42e7-99d7-3a38640957ef" alt="Loss" style="width: 30%; margin: 0 5px;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/a4c49a41-5483-480a-a105-cb83dcaf4ec8" alt="Confusion Matrix" style="width: 30%; margin: 0 5px;">
</div>

### ResNet101:
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/a033cb74-3996-4fa0-8da1-17e20001ecd9" alt="Accuracy" style="width: 30%; margin: 0 5px;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/c912ea2d-0756-4cc4-aadb-1024a2595bc0" alt="Loss" style="width: 30%; margin: 0 5px;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/872c679b-499d-4f55-8f3e-417c17fc23cb" alt="Confusion Matrix" style="width: 30%; margin: 0 5px;">
</div>

### ResNext50_32x4d:
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/340aa219-e246-4dc9-aed3-034a67991ccf" alt="Accuracy" style="width: 30%; margin: 0 5px;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/0913a51a-d95c-449d-99a5-db6fcf574a66" alt="Loss" style="width: 30%; margin: 0 5px;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/f051d9b9-1e6f-4642-9b6f-e669ec65d81f" alt="Confusion Matrix" style="width: 30%; margin: 0 5px;">
</div>

### VGG16:
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/58a32cdc-2532-42dc-8296-0dff2d07ea97" alt="Accuracy" style="width: 30%; margin: 0 5px;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/fe6102d1-139b-4a9e-a364-268af6ec23d7" alt="Loss" style="width: 30%; margin: 0 5px;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/b42c4e78-2bc2-44e6-a265-f4e8c5ee260b" alt="Confusion Matrix" style="width: 30%; margin: 0 5px;">
</div>

### VGG19:
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/a5b07571-9552-441d-a9b2-4e789f05bd97" alt="Accuracy" style="width: 30%; margin: 0 5px;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/94e395b2-813b-430d-b81e-d60120032cb0" alt="Loss" style="width: 30%; margin: 0 5px;">
    <img src="https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/e296bbed-7aaf-4149-84eb-e1cd7db0d57e" alt="Confusion Matrix" style="width: 30%; margin: 0 5px;">
</div>






