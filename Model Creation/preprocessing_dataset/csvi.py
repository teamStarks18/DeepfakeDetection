import os
import csv

# Define the folder path
folder_path = "/Users/advaithsajeev/Desktop/Intel hackthon/Final_Dataset"

# Define the subfolder names
class_0_folder = "Class_0"
class_1_folder = "Class_1"

# Define the label mappings
label_map = {
    class_0_folder: "REAL",
    class_1_folder: "FAKE"
}

# Define the output CSV file
output_csv_file = "output.csv"

# Open the CSV file for writing
with open(output_csv_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the header
    csv_writer.writerow(['Video Name', 'Label'])
    
    # Iterate through each subfolder
    for subfolder in [class_0_folder, class_1_folder]:
        subfolder_path = os.path.join(folder_path, subfolder)
        # Iterate through files in the subfolder
        for file_name in os.listdir(subfolder_path):
            # Extract video name without extension and add .mp4 extension
            video_name = os.path.splitext(file_name)[0] + ".mp4"
            # Write the video name and corresponding label to the CSV
            csv_writer.writerow([video_name, label_map[subfolder]])
