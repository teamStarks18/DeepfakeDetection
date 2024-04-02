# Deepfake Detection Web Dashboard

## Overview

Welcome to our Deepfake Detection Web Dashboard – a powerful tool designed to analyze uploaded videos using a list of tested and verified deepfake detectors. Each detector utilizes distinct techniques, having its own strengths and weaknesses. The results from these detectors are then intelligently aggregated by an aggregate model, which gives adequate weights to each model. The aggregator is finely tuned based on their historical performance of these detectors, as observed through a custom prepared dataset.


<div align="center">
  <img src="https://github.com/teamStarks18/DeepfakeDetection/blob/main/images/1.jpg" alt="1" width="200" height="155"/>
  <img src="https://github.com/teamStarks18/DeepfakeDetection/blob/main/images/2.jpg" alt="2" width="200" height="155"/>
  <img src="https://github.com/teamStarks18/DeepfakeDetection/blob/main/images/3.jpg" alt="3" width="200" height="155"/>
</div>

## Problem Statement
In the fast-evolving tech landscape, the widespread use of deepfakes, driven by advanced artificial intelligence, raises serious concerns for global information reliability 
and democratic processes. This multifaceted challenge 
includes a growing disinformation crisis, political 
manipulation threats, privacy violations, security risks, 
legal ambiguities, and difficulties in content verification. 
Our team urgently focuses on developing advanced 
detection mechanisms to counter deepfakes, preserving 
information integrity, safeguarding democracy, and 
restoring trust in our digital world. Immediate action is 
crucial to shield individuals, nations, and democratic 
processes from the escalating risks posed by deepfakes.


## Demo
This is the demo of website


https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/3e43abb9-f411-4810-861b-275f1f03f95e








## Key Features


- **Multiple Detectors:**
  - Incorporating various detectors, each trained on distinct datasets and employing diverse means of input processing or architectural designs, ensures the mitigation of individual model drawbacks and enhances generalizability. Additionally, this approach reduces vulnerability to adversarial attacks targeting specific detector weaknesses.

- **Scalability and Future-Proofing:**
  - Named "Infinite Ensemble," our system seamlessly integrates an indefinite number of detectors, with the aggregator adaptable to accommodate them. Given the dynamic nature of deepfake detection research, with newer, more proficient detectors emerging regularly, our system serves as a robust platform for testing and validating evolving detector technologies.

- **API for Everyone:**
  -  Once hosted, anyone can use our API seamlessly and they can be easily employed in their applications allowing everyone to use these detectors.

- **Parallel Processing:**
  - Addressing concerns regarding computational overhead during the system's conceptualization, we implement efficient face cropping libraries like Mediapipe and leverage parallel processing techniques. This ensures simultaneous processing of videos by multiple models, optimizing computational efficiency.

- **Aggregate Model:**
  - We have devised an aggregate model trained on individual model performances using a custom dataset. This model effectively consolidates results from all detectors, considering their respective accuracies and resolving conflicting outputs.

- **Human-in-the-Loop System:**
  - Facilitates expert analysis of stored video inputs and outputs, fostering continuous model performance enhancement, ongoing model maintenance through real-world data analysis and refinement, and efficient generation of historical performance data for new models by leveraging the models to predict outputs from a custom dataset.

### Intel AI analytics toolkit in Data preprocessing
![download](https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/095fd0ca-d4f9-424e-b366-6cfeb7f60b6e)


In our deepfake detector project, the utilization of Intel AI Analytics Toolkit, particularly Intel-optimized Pandas and NumPy, significantly bolstered our data preprocessing phase. These optimized tools enabled efficient vectorization of operations, allowing us to process large datasets swiftly. By harnessing Intel's optimizations tailored for their architectures, we expedited data cleaning, manipulation, and feature engineering tasks. As a result, we successfully streamlined our preprocessing pipeline, ensuring timely completion of the project while maintaining high performance and accuracy.

### Intel AI analytics toolkit in Model Training
Leveraging optimized versions of PyTorch from Intel, in conjunction with the Intel Developer Cloud, significantly accelerated our training process. This enabled us to efficiently train multiple models on a sufficiently large dataset and successfully complete the project within the specified deadline.

## Data Flow

![hi1 drawio](https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/b714aadc-d592-4f9b-bc67-d294357912ea)




## Usage
More about this has been discussed [here](https://github.com/teamStarks18/DeepfakeDetection/blob/main/App/Readme.md)



## Creating the AI model
More about this has been discussed [here](https://github.com/teamStarks18/DeepfakeDetection/blob/main/Model%20Creation/ReadME.md)



## Future Scopes

- **Multi-Face Handling:**
  - In the final development stages, we aim to incorporate the capability to handle videos with multiple faces. Although this may marginally increase computational time, it enhances the software's adaptability to diverse input scenarios.

- **Improving Accuracies of Individual Models:**
   - We aim to use more advanced architectures and bigger input sizes and sequence length for training. As of now, the models have been trained on smaller frame size(100- 150 pixels) and sequence lenght varying from 10 - 30. Larger images helps to capture more fine details in the frames and longer sequnces helps to capture more spatio - temporal data.

- **Expanding Dataset:**
  - With the limited time constraint for the development of the project , we had to settle down for open source datasets, but these tend to be of lower quality than SOTA deepfakes that are posed to create trouble. We aim to create a custom dataset on our own which covers broader types of deepfakes for improving generalizability.

## Contributors
- https://github.com/maheXh : AI dev
- https://github.com/kishorekuttalamr : AI dev
- https://github.com/Advaith-Sajeev : AI dev
- https://github.com/sharathcx : Full Stack dev

## Conclusion

In conclusion, our Deepfake Detection Web Dashboard represents a vital step towards addressing the growing concerns surrounding deepfake technology. By integrating multiple detectors, leveraging parallel processing techniques, and employing an aggregate model, we've developed a robust system capable of accurately identifying manipulated content. The scalability of our platform ensures its relevance in the face of evolving detection technologies, while our commitment to improving model accuracies and expanding our dataset underscores our dedication to staying ahead of emerging threats.

Moreover, the utilization of Intel AI Analytics Toolkit has significantly enhanced our data preprocessing and model training phases, enabling us to achieve high performance and accuracy within project timelines. With future plans to handle multi-face videos, enhance individual model accuracies, and expand our dataset, we're poised to further strengthen our system's capabilities and improve its effectiveness in combating deepfake threats.

We extend our gratitude to all contributors involved in this project, whose expertise and dedication have been instrumental in its success. Through collaborative efforts, we remain committed to preserving information integrity, safeguarding democratic processes, and restoring trust in the digital realm.
