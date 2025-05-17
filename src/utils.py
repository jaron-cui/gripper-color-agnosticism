import json

import cv2
import decord
import numpy as np


def load_video_frames(video_path: str) -> np.ndarray:
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    frames = []
    for i in range(len(vr)):
        frame = vr[i].asnumpy()
        frames.append(frame)

    return np.stack(frames, axis=0)


def load_gripper_values(reference_labels_path: str) -> np.ndarray:
    with open(reference_labels_path, 'r') as file:
        data = json.load(file)
    return np.array([entry['gripper'] for entry in data.values()])


def save_video(frames: np.ndarray, output_path: str, fps: float = 30.00):
    video = cv2.VideoWriter(output_path, -1, fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    video.release()
