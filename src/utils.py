import json

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
