import os
import cv2
import argparse
import mediapipe as mp

def crop_faces_in_video(input_path, output_path, face_size=(128, 128), extra_region=30):
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Iterate through each file in the input folder
    for filename in os.listdir(input_path):
        # Check if the file is a video
        if filename.endswith('.mp4'):
            # Construct the full path of the video file
            video_path = os.path.join(input_path, filename)
            
            # Read the video
            video_capture = cv2.VideoCapture(video_path)
            
            # Create output folder if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Get some properties of the video
            frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video_capture.get(cv2.CAP_PROP_FPS))
            
            # Process each frame in the video
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                
                # Convert the frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces in the frame
                results = face_detection.process(frame_rgb)
                
                # If faces are detected, crop and save them
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        
                        # Calculate coordinates of the enlarged bounding box
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                        x -= extra_region
                        y -= extra_region
                        w += 2 * extra_region
                        h += 2 * extra_region
                        
                        # Ensure the coordinates are within the image boundaries
                        x = max(0, x)
                        y = max(0, y)
                        w = min(iw - x, w)
                        h = min(ih - y, h)
                        
                        # Crop the enlarged bounding box
                        face_crop = frame[y:y+h, x:x+w]
                        
                        # Resize the cropped face to the desired size
                        face_crop_resized = cv2.resize(face_crop, face_size, interpolation=cv2.INTER_AREA)
                        
                        # Generate output filename
                        output_filename = os.path.splitext(filename)[0] + f'_face_{int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))}.jpg'
                        output_filepath = os.path.join(output_path, output_filename)
                        
                        # Save the cropped and resized face
                        cv2.imwrite(output_filepath, face_crop_resized)
            
            # Release resources
            video_capture.release()
            
            print(f"Faces cropped from video {filename} and saved in {output_path}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Crop faces from videos.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input folder containing videos')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output folder to save cropped faces')
    parser.add_argument('-e', '--extra', type=int, default=30, help='Extra region around the detected face')
    args = parser.parse_args()

    # Crop faces in videos
    crop_faces_in_video(args.input, args.output, extra_region=args.extra)
