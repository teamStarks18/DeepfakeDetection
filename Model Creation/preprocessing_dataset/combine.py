import os
import cv2
import argparse
from collections import defaultdict

def combine_frames_to_video(input_path, output_path):
    # Iterate through each file in the input folder
    video_files = defaultdict(list)

    for filename in os.listdir(input_path):
        if filename.endswith('.jpg'):
            video_name = '_'.join(filename.split('_')[:3])  # Extract video name
            video_files[video_name].append(filename)

    for video_name, frames in video_files.items():
        # Sort frames based on their sequence number
        frames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # Create video writer
        output_filename = f'{video_name}.mp4'
        output_filepath = os.path.join(output_path, output_filename)
        frame = cv2.imread(os.path.join(input_path, frames[0]))
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filepath, fourcc, 30, (width, height))

        # Combine frames into video
        for frame_name in frames:
            frame = cv2.imread(os.path.join(input_path, frame_name))
            out.write(frame)

        # Release resources
        out.release()
        print(f"Video created: {output_filepath}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Combine frames into videos.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input folder containing frames')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output folder to save videos')
    args = parser.parse_args()

    # Combine frames into videos
    combine_frames_to_video(args.input, args.output)
