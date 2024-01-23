import cv2
import numpy as np
from fitz import fitz
import logging

def is_frame_different(frame1, frame2, threshold=0.9):
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    abs_diff = cv2.absdiff(gray_frame1, gray_frame2)
    mean_abs_diff = np.mean(abs_diff)
    return mean_abs_diff > 1

def convert_millis_to_seconds(millis):
    seconds = int(millis / 1000)
    return seconds


def slide_chunking(video_path, slides_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video file.")

    slide_num = -1
    with fitz.open(slides_path, filetype="pdf") as doc:
        slide_num = len(doc)

    # Initialize list to store tuples (text, start time, end time)
    slide_chunks = []

    ret, previous_frame = cap.read()
    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) * 10)

    i = 0
    while i <= slide_num:
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
            # cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.imshow('Original Frame', frame)
            with fitz.open(slides_path, filetype="pdf") as doc:
                if i < slide_num:
                    text = doc[i].get_text()
                    # Add tuple (text, start time, end time) to the list
                    slide_chunks.append((text, timestamp, None))
                    print("the current slide chunk is : ", slide_chunks[i])
            i += 1
        previous_frame = frame.copy()

    cap.release()
    cv2.destroyAllWindows()

    # Set each tuple's end time to be the next tuple's start time
    for i in range(len(slide_chunks)-1):
        next_frame_tuple = slide_chunks[i + 1]
        slide_chunks[i] = (slide_chunks[i][0], slide_chunks[i][1], next_frame_tuple[1])

    # Remove last element of frame_list, because last slide of most lectures is consolidation/review therefore useless
    last = slide_chunks.pop()
    print(last)
    return slide_chunks

if __name__ == "__main__":
    chunks = slide_chunking("sample_data/lecture.mp4", "sample_data/hai_lecture_slides.pdf")