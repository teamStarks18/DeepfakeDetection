import cv2
import os
import random
import mediapipe as mp
import shutil


def extract_frames_and_combine_to_video(
    video_path, output_path, num_frames=10, crop_size_factor=1.5, target_size=(256, 256)
):
    # Create temporary directory for frames extraction
    temp_output_dir = "temp_frames"

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.9999999)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return None

    # Initialize video writer
    output_filename = "output_video.mp4"
    output_filepath = os.path.join(output_path, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filepath, fourcc, 30, target_size)

    frame_count = 0
    frames_with_faces = 0

    # Read until specified number of frames with faces are found
    while frames_with_faces < num_frames:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Convert the frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    ih, iw, _ = frame.shape
                    bboxC = detection.location_data.relative_bounding_box
                    x, y, w, h = (
                        int(bboxC.xmin * iw),
                        int(bboxC.ymin * ih),
                        int(bboxC.width * iw),
                        int(bboxC.height * ih),
                    )
                    # Calculate expanded bounding box coordinates
                    cx, cy = (
                        x + w // 2,
                        y + h // 2,
                    )  # Center of the original face bounding box
                    nw, nh = int(w * crop_size_factor), int(
                        h * crop_size_factor
                    )  # New width and height
                    nx, ny = max(0, cx - nw // 2), max(
                        0, cy - nh // 2
                    )  # New top-left corner
                    nx2, ny2 = min(iw, nx + nw), min(
                        ih, ny + nh
                    )  # New bottom-right corner

                    # Ensure the coordinates are within the frame boundaries
                    nx, ny, nx2, ny2 = (
                        max(nx, 0),
                        max(ny, 0),
                        min(nx2, iw),
                        min(ny2, ih),
                    )

                    # Crop the face region from the frame
                    face_crop = frame[ny:ny2, nx:nx2]

                    # Resize the cropped face region to the target size
                    resized_face_crop = cv2.resize(face_crop, target_size)

                    # Write the resized face crop to the output video
                    out.write(resized_face_crop)

                    # Increment frames with faces count
                    frames_with_faces += 1

                    if frames_with_faces >= num_frames:
                        break

        else:
            break

    # Release resources
    cap.release()
    out.release()

    print(f"{frames_with_faces} frames with faces extracted from '{video_path}'")

    return output_filepath


# Example usage:
# input_video = r"C:\Users\mahes\ML\dataset\data unziped\DFDC\dfdc_train_part_0\kdzirwrnyk.mp4"
# output_directory = "output_videos_real"
# num_frames = 150
# crop_size_factor = 1.5
# target_size = (256, 256)
# new_video_path = extract_frames_and_combine_to_video(input_video, output_directory, num_frames, crop_size_factor, target_size)
# print("Newly created video path:", new_video_path)
