from .eval import Evaluator
from . import datasets
from . import metrics
from . import plotting
from . import utils
def calculate_ranges(total_frames, n, overlap_ratio, frames_per_range = 75):



        overlap_frames = int(frames_per_range * overlap_ratio)

        ranges = []

        for i in range(n):
            start_frame = i * (frames_per_range - overlap_frames) + 1
            end_frame = start_frame + frames_per_range - 1


            if i == n - 1:
                end_frame = total_frames

            ranges.append((start_frame, end_frame))

        return ranges
