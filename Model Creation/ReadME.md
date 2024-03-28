link to the drive containing the trained models: https://drive.google.com/drive/folders/1o4lNbL9odOtQoXiELppH3z4IUuAV30fn?usp=sharing 
# Model Creation 
This directory contains all the codes for the Detectors, Inference, and Training.

## Dataset
The following datasets were utilized in training our model:
- [FaceForensics](https://github.com/ondyari/FaceForensics)
- [CelebDF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/data)
<img src="https://github.com/teamStarks18/DeepfakeDetection/blob/main/images/dataseticon.png" alt="1" width="200" height="155" align="right"/>

## Custom Dataset
Randomly selected an equal number of videos from all the mentioned datasets. Various augmentation techniques were implemented to increase the number of samples in the dataset.

## Preprocessing
All the programs used for data preprocessing and augmentation are located under the [Preprocessing Dataset](https://github.com/teamStarks18/DeepfakeDetection/tree/main/Model%20Creation/preprocessing_dataset) directory. The approach involved extracting all frames from the video, using Mediapipe to detect faces in each frame, adding required padding to the region of interest to gather more information from around the face and the face itself while ignoring the rest of the video. These cropped images are then combined to form the preprocessed video. Techniques such as horizontal flipping were used to increase the number of inputs.
  
 

 
