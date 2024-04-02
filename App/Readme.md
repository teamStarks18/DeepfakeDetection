
![transition drawio](https://github.com/teamStarks18/DeepfakeDetection/assets/161623545/b362b91d-873a-481f-8f89-cc7df0479e05)


## Tech Stack

- FrontEnd - React
- BackEnd -  FastAPI, Python


## Deployment

To deploy this project run
### Clone the Repository

```bash
  git clone https://github.com/teamStarks18/DeepfakeDetection
```

## FrontEnd
The frontend part has been hosted [here](https://github.com/teamStarks18/project-repo)

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

