import json

import cv2
import decord
import numpy as np
from scipy.ndimage import binary_dilation


def main():
    reference_video_path = '../reference_data/compressed_video_h264.mp4'
    reference_labels_path = '../reference_data/labels.json'
    output_path = '../augmentation_masks/augmentation_masks.npy'
    generate_augmentation_masks(reference_video_path, reference_labels_path, output_path)


def generate_augmentation_masks(reference_video_path: str, reference_labels_path: str, output_path: str):
    frames = load_video_frames(reference_video_path)
    masks = np.stack([mask_blue_tpu_pixels(frame) for frame in frames])
    gripper_values = load_gripper_values(reference_labels_path)
    np.save(output_path, {
        'gripper_values': gripper_values,
        'masks': masks
    })


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


def mask_blue_tpu_pixels(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # mask by blue color
    is_blue_hue = (hsv[:, :, 0] < 110) & (hsv[:, :, 0] > 80)
    is_saturated = (hsv[:, :, 1] > 60)
    is_bright = (hsv[:, :, 2] > 140)
    gripper_mask = is_blue_hue & is_saturated & is_bright

    # expand mask by 2 pixel margin
    pixels = 2
    structure = np.ones((2 * pixels + 1, 2 * pixels + 1), dtype=bool)
    gripper_mask = binary_dilation(gripper_mask.reshape(hsv[:, :, 0].shape), structure)

    # reflect mask for symmetry
    gripper_mask = gripper_mask | np.flip(gripper_mask, axis=1)

    return gripper_mask


if __name__ == '__main__':
    main()
