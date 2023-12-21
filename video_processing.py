import cv2
import numpy as np

def is_frame_different(frame1, frame2, threshold=0.9):
    # Convert frames to grayscale
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    abs_diff = cv2.absdiff(gray_frame1, gray_frame2)
    # Calculate the mean of absolute differences
    mean_abs_diff = np.mean(abs_diff)
    return mean_abs_diff > 1

# Initialize list to store tuples (frame, start time, end time)
different_frames_list = []
video_path = 'sample_data/lecture.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Read the first frame to initialize previous_frame
ret, previous_frame = cap.read()

if not ret:
    print("End of video.")
    exit()

# Loop to read and process every n frames
frame_interval = 300
while True:
    # Read the next frame
    for _ in range(frame_interval - 1):
        ret, _ = cap.read()  # Skip frames
        if not ret:
            print("End of video.")
            break

    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Get the current position in milliseconds
    current_time_millis = cap.get(cv2.CAP_PROP_POS_MSEC)

    # Convert milliseconds to readable time format (assuming 30 frames per second)
    current_time_sec = current_time_millis / 1000.0
    minutes = int(current_time_sec / 60)
    seconds = int(current_time_sec % 60)
    timestamp = f"{minutes:02d}:{seconds:02d}"

    if is_frame_different(frame, previous_frame):
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Original Frame', frame)

        # Add tuple (frame, start time, end time) to the list
        different_frames_list.append((frame.copy(), timestamp, None))

    # Set the current frame as the previous frame for the next iteration
    previous_frame = frame.copy()

    # Break the loop if the user presses 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Update end time for the last frame in the list
if different_frames_list:
    # Calculate the duration of the video using the frame rate and total frames
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = total_frames / frame_rate
    # Convert seconds to minutes and seconds
    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)

    # Format the result as "minutes:seconds"
    formatted_time = f"{minutes}:{seconds:02d}"

    # Set the end time of the last frame to the video duration
    different_frames_list[-1] = (different_frames_list[-1][0], different_frames_list[-1][1], formatted_time)

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

# Print the list of different frames and their timestamps
print("Different Frames List:")
for i in range(len(different_frames_list)-1):
    frame_tuple = different_frames_list[i]
    next_frame_tuple = different_frames_list[i + 1]
    frame_tuple = (frame_tuple[0], frame_tuple[1], next_frame_tuple[1])
    print(f"Start Time: {frame_tuple[1]}, End Time: {frame_tuple[2]}")

# Print the last frame separately (no next frame)
last_frame_tuple = different_frames_list[-1]
print(f"Start Time: {last_frame_tuple[1]}, End Time: {last_frame_tuple[2]}")
