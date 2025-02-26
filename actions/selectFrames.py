import numpy as np
import matplotlib.pyplot as plt

def find_ed_es(tensor):
    """
    Finds End-Diastole (ED) and End-Systole (ES) frames.

    Args:
        tensor (numpy.ndarray): Input tensor of shape (num_frames, 384, 384).

    Returns:
        dict: ED & ES frame indices and areas.
    """
    max_area, min_area = 0, float('inf')
    ed_frame, es_frame = -1, -1

    for i, frame in enumerate(tensor):  # No need for tensor[:, 1, :, :]
        area = np.count_nonzero(frame)  # Count non-zero pixels

        if area > max_area:
            max_area = area
            ed_frame = i

        if 0 < area < min_area:  # Ignore empty frames
            min_area = area
            es_frame = i

    return {
        "ED_frame": ed_frame, "ED_area": max_area,
        "ES_frame": es_frame, "ES_area": min_area
    }


def runSelectFrames(Tensor2ch, Tensor4ch):
    result2ch = find_ed_es(Tensor2ch)
    result4ch = find_ed_es(Tensor4ch)

    print(f"2 Chamber - End-Diastole (ED) - Frame: {result2ch['ED_frame']}, Area: {result2ch['ED_area']}")
    print(f"2 Chamber - End-Systole (ES) - Frame: {result2ch['ES_frame']}, Area: {result2ch['ES_area']}")

    print(f"4 Chamber - End-Diastole (ED) - Frame: {result4ch['ED_frame']}, Area: {result4ch['ED_area']}")
    print(f"4 Chamber - End-Systole (ES) - Frame: {result4ch['ES_frame']}, Area: {result4ch['ES_area']}")


    return (result2ch, result4ch)


