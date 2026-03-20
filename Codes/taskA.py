import cv2
import numpy as np

# Task 1: Check All Frames for Day/Night Detection and Adjust Brightness for Nighttime Videos
def adjust_brightness_if_night_all_frames(video_path, output_path, save_directory, threshold=100):
    cap = cv2.VideoCapture(video_path)  # Open the video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get video height
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # Output video file

    total_brightness = 0
    total_frames = 0

    # Calculate average brightness across all frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        total_brightness += np.mean(gray)  # Add frame's brightness to the total
        total_frames += 1  # Increment frame count

    avg_brightness_video = total_brightness / total_frames if total_frames > 0 else 0

    # Determine if the video is nighttime
    is_night = avg_brightness_video < threshold

    # Reset the video capture to process frames again
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # If it's nighttime, adjust brightness
        if is_night:
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=50)  # Increase brightness

        out.write(frame)  # Write the processed frame to the output video

    cap.release()  # Release the video capture object
    out.release()  # Release the video writer object
    print(f"Processing Task 1 complete. Video saved to {output_path}")
    print("Daytime video" if not is_night else "Nighttime video (brightness adjusted)")

# Task 2: Blur Faces in the Video
def blur_faces(video_path, output_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(video_path)  # Open the video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get video height
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # Output video file

    while cap.isOpened():
        ret, frame = cap.read()  # Read each frame
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces
        
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]  # Extract face region
            # Apply Gaussian blur to the face region
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(face_region, (51, 51), 30)
        
        out.write(frame)  # Write the processed frame to output video
    
    cap.release()  # Release the video capture object
    out.release()  # Release the video writer object
    print(f"Processing Task 2:Blur faces complete. Video saved to {output_path}")

# Task 3: Overlay Talking Video on Top Left with Border and Adjusted Position
def overlay_video(main_video_path, overlay_video_path, output_path, position=(100, 100), size=(300, 190), border_color=(0, 0, 0), border_thickness=5):
    cap_main = cv2.VideoCapture(main_video_path)  # Open the main video
    cap_overlay = cv2.VideoCapture(overlay_video_path)  # Open the talking video
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    fps = int(cap_main.get(cv2.CAP_PROP_FPS))  # Get frames per second of main video
    width = int(cap_main.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get width of the main video
    height = int(cap_main.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get height of the main video
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # Output video file

    while cap_main.isOpened() and cap_overlay.isOpened():
        ret_main, main_frame = cap_main.read()  # Read frame from main video
        ret_overlay, overlay_frame = cap_overlay.read()  # Read frame from overlay video
        
        if not ret_main:
            break
        
        if ret_overlay:
            # Resize the overlay video frame to fit the top-left corner
            overlay_frame = cv2.resize(overlay_frame, size)
            
            # Add a border around the overlay frame
            overlay_frame = cv2.copyMakeBorder(overlay_frame, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=border_color)
            
            # Update the size to include the border
            new_size = (overlay_frame.shape[1], overlay_frame.shape[0])
            
            # Adjust position to move closer to the center
            x, y = position
            x = min(x, width - new_size[0])  # Ensure the overlay doesn't go out of bounds
            y = min(y, height - new_size[1])  # Ensure the overlay doesn't go out of bounds
            
            # Overlay the resized video with border on the main video frame
            main_frame[y:y+new_size[1], x:x+new_size[0]] = overlay_frame
        
        out.write(main_frame)  # Write the processed frame to output video
    
    cap_main.release()  # Release main video capture object
    cap_overlay.release()  # Release overlay video capture object
    out.release()  # Release output video writer object
    print(f"Processing Task 3:Overlaying video complete. Video saved to {output_path}")

# Task 4: Overlay Watermarks on Video
def overlay_watermark(input_video_path, watermark1_image_path, watermark2_image_path, output_video_path, switch_time=5):
    cap = cv2.VideoCapture(input_video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    watermark1 = cv2.imread(watermark1_image_path, cv2.IMREAD_UNCHANGED)
    watermark2 = cv2.imread(watermark2_image_path, cv2.IMREAD_UNCHANGED)

    mask1 = cv2.cvtColor(watermark1, cv2.COLOR_BGR2GRAY)
    _, binary_mask1 = cv2.threshold(mask1, 1, 255, cv2.THRESH_BINARY)

    mask2 = cv2.cvtColor(watermark2, cv2.COLOR_BGR2GRAY)
    _, binary_mask2 = cv2.threshold(mask2, 1, 255, cv2.THRESH_BINARY)

    current_time = 0
    switch_frame_count = switch_time * fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_time < switch_frame_count:
            watermark = watermark1
            binary_mask = binary_mask1
        else:
            watermark = watermark2
            binary_mask = binary_mask2
        
        for c in range(0, 3):
            frame[:, :, c] = np.where(binary_mask == 255, watermark[:, :, c], frame[:, :, c])

        out.write(frame)

        current_time += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing Task 4:Adding watermark complete. Video saved to {output_video_path}")

# Task 5: Add Endscreen Video After Main Video 
def add_endscreen(input_video_path, endscreen_video_path, output_video_path):
    cap_main = cv2.VideoCapture(input_video_path)

    frame_width = int(cap_main.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_main.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_main.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap_main.isOpened():
        ret, frame = cap_main.read()
        if not ret:
            break
        out.write(frame)

    cap_endscreen = cv2.VideoCapture(endscreen_video_path)

    while cap_endscreen.isOpened():
        ret, frame = cap_endscreen.read()
        if not ret:
            break
        
        frame_resized = cv2.resize(frame, (frame_width, frame_height))

        out.write(frame_resized)

    cap_main.release()
    cap_endscreen.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing Task 5:Add endscreen complete. Video saved to {output_video_path}")

# Main function to take video_path1 as input
def main():
    video_path1 = input("Enter the path to the video file: ")
    save_directory = r"C:\Users\User\Desktop\CSC 2024 Assignment\Recorded Videos (4)"
    
    
        
    output_path1 = "adjusted_video.mp4"
    adjust_brightness_if_night_all_frames(video_path1, output_path1, save_directory)
    

    output_path2 = "blurred_video.mp4"
    blur_faces(output_path1, output_path2)

    overlay_video_path = "talking.mp4"
    output_path3 = "overlaid_video.mp4"
    overlay_video(output_path2, overlay_video_path, output_path3)

    watermark1_image_path = "watermark1.png"
    watermark2_image_path = "watermark2.png"
    output_path4 = "watermarked_video.mp4"
    overlay_watermark(output_path3, watermark1_image_path, watermark2_image_path, output_path4)

    endscreen_video_path = "endscreen.mp4"
    output_path5 = "final_video.mp4"
    add_endscreen(output_path4, endscreen_video_path, output_path5)

if __name__ == "__main__":
    main()
