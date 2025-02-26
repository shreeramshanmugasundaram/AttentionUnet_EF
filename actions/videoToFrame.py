import cv2
import numpy as np
import pydicom
import os
from pydicom.pixel_data_handlers.util import apply_voi_lut

def center_crop_resize(image, target_size=(384, 384)):
    """Crop the center and resize while keeping the important information."""
    h, w = image.shape  # Get original dimensions

    # Find the minimum center crop size
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2

    # Crop to square
    cropped = image[start_y:start_y+min_dim, start_x:start_x+min_dim]

    # Resize to target size
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    
    return resized

def load_dicom_frames(dicom_folder):
    """Load all frames from a series of DICOM images in a folder (grayscale, 384x384)."""
    dicom_files = sorted([f for f in os.listdir(dicom_folder) if f.endswith(".dcm")])
    frames = []

    for file in dicom_files:
        dicom_path = os.path.join(dicom_folder, file)
        dicom_image = pydicom.dcmread(dicom_path)
        frame = dicom_image.pixel_array

        # Apply VOI LUT if available (better contrast)
        if "VOILUTFunction" in dicom_image:
            frame = apply_voi_lut(frame, dicom_image)

        # Normalize pixel values to 0-255 range
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255.0
        frame = frame.astype(np.uint8)

        # Crop & resize to 384x384
        frame_resized = center_crop_resize(frame)

        frames.append(frame_resized)

    tensor_array = np.array(frames, dtype=np.uint8)  # Shape: (num_frames, 384, 384)
    tensor_array = np.expand_dims(tensor_array, axis=-1)  # Shape: (num_frames, 384, 384, 1)

    return tensor_array

def video_to_tensor(video_path):
    """Load frames from a standard video file, convert to grayscale, and resize to 384x384."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video FPS: {fps:.2f} frames per second")
    print(f"Video Codec: {codec}")
    print(f"Video Duration: {duration:.2f} seconds ({int(duration//60)} min {int(duration%60)} sec)")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no more frames

        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crop & resize to 384x384
        frame_resized = center_crop_resize(frame_gray)

        frames.append(frame_resized)

    cap.release()

    tensor_array = np.array(frames, dtype=np.uint8)  # Shape: (num_frames, 384, 384)
    tensor_array = np.expand_dims(tensor_array, axis=-1)  # Convert to (num_frames, 384, 384, 1)

    print(f"Final Tensor Shape: {tensor_array.shape}")
    
    return tensor_array

def convert_to_tensor(input_path):
    """Automatically detect input type (DICOM or standard video) and convert to 384x384 grayscale tensor with shape (num_frames, 384, 384, 1)."""
    if os.path.isdir(input_path):  # If input is a folder, assume it's a DICOM series
        print("Detected DICOM format.")
        tensor = load_dicom_frames(input_path)
    else:  # Otherwise, assume it's a standard video file
        print("Detected standard video format.")
        tensor = video_to_tensor(input_path)
    
    return tensor

# Example Usage:
# For standard videos: convert_to_tensor("input_video.avi")
# For DICOM: convert_to_tensor("dicom_folder/")
# tensor = convert_to_tensor("input_video.avi")  # Change to "dicom_folder/" for DICOM

def runVideoToFrame(input_video_path_2ch,input_video_path_4ch):
    Tensor2ch = convert_to_tensor(input_video_path_2ch )
    Tensor4ch = convert_to_tensor(input_video_path_4ch)

    return (Tensor2ch, Tensor4ch)