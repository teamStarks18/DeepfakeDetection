from fastapi import FastAPI, Body, UploadFile, File, HTTPException, Form
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from engine import Detector
from fastapi.responses import RedirectResponse

# from main.database import userdata, filedata
# from main.engine import Detector
import uvicorn
import json
from fastapi.responses import FileResponse
import time

# from pathlib import Path
# import matplotlib.pyplot as plt
# import seaborn as sns


# def create_plot(result):
#     # Extracting model names and corresponding accuracies
#     models = [f'Model {i + 1}' for i in range(len(result) - 1)]  # -1 to exclude the aggregated_probability key
#     accuracies = [result[key] for key in result.keys() if 'model' in key]
#
#     # Extracting aggregated accuracy
#     aggregated_accuracy = result['aggregated_probability']
#
#     # Set a custom color palette for vibrant colors
#     colors = sns.color_palette('husl', n_colors=len(models))
#
#     # Set a dark grid style for better contrast
#     sns.set_style("darkgrid")
#
#     # Plotting the data
#     fig, ax = plt.subplots(figsize=(12, 8))
#
#     # Calculate the starting point for the y-axis
#     min_y = min(accuracies) - 30
#
#     # Set y-axis limits
#     ax.set_ylim(min_y, 100)  # Start from 30 less than the least value
#
#     bars = sns.barplot(x=models, y=accuracies, palette=colors, ax=ax)
#
#     # Adding text labels for each bar
#     for bar, model, color in zip(bars.patches, models, colors):
#         yval = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{round(yval, 2)}%', ha='center', va='bottom',
#                 fontsize=12, color='black', fontweight='bold')
#
#         # Add model names inside the bars with a different color
#         ax.text(bar.get_x() + bar.get_width() / 2, yval / 2, model, ha='center', va='center', fontsize=10, color=color,
#                 rotation=45)
#
#     # Adding aggregated accuracy at the bottom with a different color
#     ax.text(0.5, -0.15, f'Aggregated Accuracy: {round(aggregated_accuracy, 2)}%', transform=ax.transAxes, ha='center',
#             va='bottom', fontsize=35, color='red', fontweight='bold')
#
#     # Adding title and labels
#     plt.title('Model Accuracies', fontsize=18, fontweight='bold')
#     plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
#
#     # Increase the markings on the y-axis
#     ax.yaxis.set_major_locator(plt.MaxNLocator(10))
#
#     # Display the plot
#     plt.tight_layout()
#     # plt.show()
#     path_main = r"C:\Users\mahes\ML\NESTRIS\main\plot.png"
#     plt.savefig(path_main)


app = FastAPI()

# Allow requests from any origin
# Allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def redirect():
    return RedirectResponse(url="https://project.webfork.tech")


# sign up
# @app.get("/signup")
# async def sign_up(name: str, username: str, password: str):
#     count = userdata.count_documents({'username': username})
#     if count != 0:
#         raise HTTPException(status_code=400, detail="Username already exists")
#     userdata.insert_one({'name': name, 'username': username, 'password': password})
#     return {"result": True}


# login
# @app.get("/login")
# async def login(username: str, password: str):
#     count = userdata.count_documents({'username': username})
#     if count == 0:
#         raise HTTPException(status_code=400, detail="Username already exists")
#     authData = userdata.find_one({"username": username})
#     if authData["password"] == password:
#         return True
#     raise HTTPException(status_code=400, detail="Username already exists")


# data should be passed in the format given below


# const data = { models: ["m1", "m2", "m3"] };
#
# fetch('http://localhost:8000/models/', {
#   method: 'POST',
#   headers: {
#     'Content-Type': 'application/json',
#   },
#   body: JSON.stringify(data),
# })
@app.post("/models")
async def get_data(file: UploadFile = File(...), models: str = Form(...)):
    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)

    path = file.filename
    print(path)

    models_list = models.split(",")
    # print(models_list)

    detector = Detector(path)

    result = detector.aggregate()
    names = list(result.keys())
    vals = list(result.values())

    # print(vals)
    # filedata.insert_one({"filename": file.filename, "result": result})
    # create_plot(result)
    # print(type(result))
    # d = {"modelName": list(result.keys()), "accuracy": list(result.values())}
    # return {"modelName": list(result.keys()), "accuracy": list(result.values())}

    d = {
        "modelName": names,
        "accuracy": [str(i) for i in vals],
    }
    print(d)
    return d


# @app.get("/get_file")
# async def get_file():
#     path = r"C:\Users\mahes\ML\NESTRIS\main\plot.png"
#     return FileResponse(path, filename="plot.png")


# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)
