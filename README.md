# Deepfake Detection Web Dashboard

## Overview

Welcome to our Deepfake Detection Web Dashboard – a powerful tool designed to analyze uploaded videos using a meticulously curated list of tested and verified deepfake detectors. Each detector in our arsenal utilizes distinct techniques, bringing its own strengths and weaknesses to the table. The results from these detectors are then intelligently aggregated by an aggregate model, finely tuned based on their historical performance, as observed through a meticulously prepared custom dataset.
<div align="center">
  <img src="https://github.com/teamStarks18/DeepfakeDetection/blob/main/images/1.jpg" alt="1" width="200" height="155"/>
  <img src="https://github.com/teamStarks18/DeepfakeDetection/blob/main/images/2.jpg" alt="2" width="200" height="155"/>
  <img src="https://github.com/teamStarks18/DeepfakeDetection/blob/main/images/3.jpg" alt="3" width="200" height="155"/>
</div>




## Key Features

- **Multiple Detectors:**
  - Incorporating various detectors, each trained on distinct datasets and employing diverse means of input processing or architectural designs, ensures the mitigation of individual model drawbacks and enhances generalizability. Additionally, this approach reduces vulnerability to adversarial attacks targeting specific detector weaknesses.

- **Scalability and Future-Proofing:**
  - Named "Infinite Ensemble," our system seamlessly integrates an indefinite number of detectors, with the aggregator adaptable to accommodate them. Given the dynamic nature of deepfake detection research, with newer, more proficient detectors emerging regularly, our system serves as a robust platform for testing and validating evolving detector technologies.

- **Parallel Processing:**
  - Addressing concerns regarding computational overhead during the system's conceptualization, we implement efficient face cropping libraries like Mediapipe and leverage parallel processing techniques. This ensures simultaneous processing of videos by multiple models, optimizing computational efficiency.

- **Multi-Face Handling:**
  - In the final development stages, we aim to incorporate the capability to handle videos with multiple faces. Although this may marginally increase computational time, it enhances the software's adaptability to diverse input scenarios.

- **Aggregate Model:**
  - We have devised an aggregate model trained on individual model performances using a custom dataset. This model effectively consolidates results from all detectors, considering their respective accuracies and resolving conflicting outputs.

- **Human-in-the-Loop System:**
  - Facilitates expert analysis of stored video inputs and outputs, fostering continuous model performance enhancement, ongoing model maintenance through real-world data analysis and refinement, and efficient generation of historical performance data for new models by leveraging the models to predict outputs from a custom dataset.


## Data Flow

![newMain](https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/c0e41252-94e3-487e-add1-b12050039cb8)


## How to Run



## Training and Testing
More about this has been discussed [here](https://github.com/teamStarks18/DeepfakeDetection/blob/main/Model%20Creation/ReadME.md)
