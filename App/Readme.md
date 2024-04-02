
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
  # Create a new virtual environment named detector_env
conda create --name detector_env

# Activate the virtual environment
conda activate detector_env

# Install PyTorch and related packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install mediapipe and other packages
pip install mediapipe opencv-python numpy pandas matplotlib scikit-learn seaborn
```
- Follow the Readme [here](https://github.com/teamStarks18/DeepfakeDetection/tree/main/App/detectors)


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

