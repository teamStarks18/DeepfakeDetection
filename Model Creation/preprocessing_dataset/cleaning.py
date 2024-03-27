import os
import random
import shutil

# Define the folder path
folder_path = "/Users/advaithsajeev/Desktop/Intel hackthon/Final_Dataset/Class_1"

# Define the number of videos to keep
videos_to_keep = 1800

# Get the list of all video files in the folder
video_files = [file for file in os.listdir(folder_path) if file.endswith(".mp4")]

# Randomly select videos to keep
videos_to_delete = random.sample(video_files, len(video_files) - videos_to_keep)

# Delete the videos
for video in videos_to_delete:
    video_path = os.path.join(folder_path, video)
    os.remove(video_path)
    print(f"Deleted: {video}")

print("Videos deleted successfully.")
