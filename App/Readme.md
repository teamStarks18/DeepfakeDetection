
# How to Run

This part explain about the functionalty of the website along with how to run the Code.


Currently, the website employs an older version of our software, in which the various detectors are slightly altered versions of the same architecture (ResNext with LSTM). They are trained on different subsets of data and have different sequence lengths as input to the models. We have experimented with different model architectures, utilized better preprocessing techniques, and developed a custom dataset prepared by us to improve the models' performance. All the resources for this are shared in the resource creation directory. In the final stages of our project, we aim to integrate our best detectors into our website.

![Screenshot from 2024-03-28 16-42-12](https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/93b1f29e-45f6-48d2-876a-bb929d538b02)
![Screenshot from 2024-03-28 16-42-58](https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/d95a2b02-3efe-413f-ba72-bf04e755f9ba)
## Tech Stack

- FrontEnd - React
- BackEnd -  FastAPI, Python



## Deployment

To deploy this project run
### Clone the Repository

```bash
  git clone https://github.com/teamStarks18/DeepfakeDetection
```

### Setting Backend

```bash
  pip install -r requirements.txt
```
- Download all the weights inside the [link](https://drive.google.com/drive/folders/1vlEqfVGyY9OehNsLkgRLReviaG3c79cU) and paste it inside directory Detector/weights.

- Download the file inside the [link](https://drive.google.com/file/d/1Vp1LIbY6LE8U6_apxhHibIbY2MsttY8K/view?usp=drive_link) and paste it inside root directory DeepfakeDetection.

### Setup NGRok
- [Download ngRok](https://ngrok.com/download)
- [Login to](https://dashboard.ngrok.com/signup)
- [Setup Authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
- [Generate a static URL](https://dashboard.ngrok.com/cloud-edge/domains)
- ngrok http --domain={summary-wildly-rodent.ngrok-free.app} 80, this is a sample Domain URL, change the URL as per yours.
- Update the app.jsx line line number 121
const response = await fetch('{https://largely-smashing-pangolin.ngrok-free.app/}models/' with the ngrok Link
- Run [app.py]() using the command 
```bash
  uvicorn app:app --reload --port 80
```
use the ngrok link to access the website.

