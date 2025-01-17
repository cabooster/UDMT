from .eval import Evaluator
from . import datasets
from . import metrics
from . import plotting
from . import utils
def calculate_ranges(total_frames, n, overlap_ratio, frames_per_range = 75):
        # frames_per_range = total_frames // n  # 每个范围的帧数

        # 计算重合帧数
        overlap_frames = int(frames_per_range * overlap_ratio)

        ranges = []  # 存储范围的起始帧和结束帧

        for i in range(n):
            start_frame = i * (frames_per_range - overlap_frames) + 1
            end_frame = start_frame + frames_per_range - 1

            # 如果是最后一个范围，调整结束帧以确保覆盖所有帧
            if i == n - 1:
                end_frame = total_frames

            ranges.append((start_frame, end_frame))

        return ranges
