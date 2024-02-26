import cv2
import numpy as np
import pytesseract
import logging
import pickle
from moviepy.editor import VideoFileClip
from alinet import chunking
from alinet import qg

with open("experiments/qg/comp3074_lecture_2.pkl", "rb") as file:
        whisper_chunks = pickle.load(file)['chunks']
chunk_pipe = chunking.Pipeline(qg.Model.BASELINE)
transcript_chunks = chunk_pipe(whisper_chunks, 2301)

def is_frame_different(frame1, frame2, threshold=0.9):
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    abs_diff = cv2.absdiff(gray_frame1, gray_frame2)
    mean_abs_diff = np.mean(abs_diff)
    return mean_abs_diff > threshold

def convert_millis_to_seconds(millis):
    seconds = int(millis / 1000)
    return seconds

def slide_chunking(video_path):
    """
    Extract slide chunks from a video based on frame differences and slide timestamps.

    Parameters:
    - video_path (str): Path to the video file.

    Returns:
    list: A list of tuples representing slide chunks with (text, start time, end time).
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video file.")

    # Initialize list to store tuples (text, start time, end time)
    slide_chunks = []

    ret, previous_frame = cap.read()
    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) * 30)

    i = 0
    while True:
        for _ in range(frame_interval - 1):
            ret, _ = cap.read()  # Skip frames
            if not ret:
                logging.info("End of video")
                break

        ret, frame = cap.read()
        if not ret:
            logging.info("End of video")
            break

        current_time_millis = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = convert_millis_to_seconds(current_time_millis)

        if is_frame_different(frame, previous_frame):
            """
            Uncomment the 2 lines below to load a window that displays the current frames and respective timestamp
            """
            # cv2.putText(frame, str(timestamp), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.imshow('Original Frame', frame)
            text = pytesseract.image_to_string(frame)
            slide_chunks.append((text, timestamp, None))
            i += 1
        previous_frame = frame.copy()

    cap.release()
    cv2.destroyAllWindows()

    # Set each tuple's end time to be the next tuple's start time
    for i in range(len(slide_chunks)-1):
        next_frame_tuple = slide_chunks[i + 1]
        slide_chunks[i] = (slide_chunks[i][0], slide_chunks[i][1], next_frame_tuple[1])

    # Remove last element of frame_list, because the last slide of most lectures is consolidation/review, therefore, useless
    slide_chunks.pop()
    return slide_chunks

def save_clips_locally(video_path, chunks, output_dir_path):
    previous_end_time = 0 
    # NOTE: The 25-second estimation below is derived from our stride length in /chunking/pipeline, 
    # also we assume words are approximately 3/4 of a token, and the average human speaks at a rate of 140 words per minute.
    stride_time = 25 

    for i, chunk in enumerate(chunks):
        start_time, end_time = chunk["timestamp"]

        # ensure stride adjustment occurs only if possible
        if i != 0 and previous_end_time >= stride_time:
            start_time -= stride_time
        
        subclip = VideoFileClip(video_path).subclip(start_time, end_time)
        subclip.write_videofile(f"{output_dir_path}/chunk{i}.mp4", codec='libx264', audio_codec='aac')
        previous_end_time = end_time 

if __name__ == "__main__":
    chunks = slide_chunking("sample_data/lecture.mp4")
