import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import cv2
import os

class TrackletVisualizer:
    def __init__(self, video_frames_path, track_path):
        # Load video frames (only .jpg files)
        self.video_frames_path = video_frames_path
        self.frames = sorted(
            [f for f in os.listdir(video_frames_path) if f.endswith(".jpg")]
        )


        self.nframes = len(self.frames)

        # Load track data (shape: [num_animals, num_frames, 2])
        self.track_data = np.load(track_path)
        self.num_animals, self.nframes_from_data, _ = self.track_data.shape


        if self.nframes != self.nframes_from_data:
            warning_message = (
                f"Warning: The number of video frames ({self.nframes}) does not match the number of frames in the track data ({self.nframes_from_data}).\n"
                "The video frames will be truncated to match the track data."
            )
            print(warning_message)
            self.frames = self.frames[:self.nframes_from_data]
            self.nframes = len(self.frames)

        # Colors for each animal
        self.colors = plt.cm.get_cmap('viridis', self.num_animals).colors

        # Initialize current frame index
        self.curr_frame = 0
        self.point_size = 100

        # Prepare figure
        self.fig = plt.figure(figsize=(15, 8))
        gs = self.fig.add_gridspec(2, 2)
        self.ax_video = self.fig.add_subplot(gs[:, 0])
        self.ax_x = self.fig.add_subplot(gs[0, 1])
        self.ax_y = self.fig.add_subplot(gs[1, 1], sharex=self.ax_x)

        # Remove axis for video
        self.ax_video.axis("off")
        plt.subplots_adjust(bottom=0.2)

        # Add slider
        self.ax_slider = self.fig.add_axes([0.1, 0.1, 0.8, 0.03], facecolor="lightgray")
        self.slider = Slider(self.ax_slider, "Frame", 0, self.nframes - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.update_frame)

        self.ax_slider_size = self.fig.add_axes([0.1, 0.05, 0.5, 0.03], facecolor="orange")
        self.slider_size = Slider(self.ax_slider_size, "Dot Size", 10, 200, valinit=self.point_size, valstep=10)
        self.slider_size.on_changed(self.update_dot_size)

        # Plot data placeholders
        self.im_video = None
        self.scat_points = None
        self.lines_x = []
        self.lines_y = []
        self.vline_x = None
        self.vline_y = None

        # Initialize plots
        self._initialize_plots()

    def _initialize_plots(self):
        # Load the first frame and display
        frame_path = os.path.join(self.video_frames_path, self.frames[self.curr_frame])
        img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        self.im_video = self.ax_video.imshow(img)

        # Plot tracks in x and y axes
        for i in range(self.num_animals):
            x_data = self.track_data[i, :, 0]
            y_data = self.track_data[i, :, 1]
            self.lines_x.append(
                self.ax_x.plot(range(self.nframes), x_data, "-", color=self.colors[i], label=f"Animal {i+1}")[0]
            )
            self.lines_y.append(
                self.ax_y.plot(range(self.nframes), y_data, "-", color=self.colors[i], label=f"Animal {i+1}")[0]
            )

        self.ax_x.set_ylabel("X Coordinate")
        self.ax_y.set_ylabel("Y Coordinate")
        self.ax_y.set_xlabel("Frame")
        # Disable legend
        self.ax_x.legend().set_visible(False)
        self.ax_y.legend().set_visible(False)
        # self.ax_x.legend(frameon=False, loc="upper right", fontsize="small")
        # self.ax_y.legend(frameon=False, loc="upper right", fontsize="small")

        # Add scatter points for the current frame
        self.scat_points = self.ax_video.scatter([], [], s=self.point_size, edgecolor="white", lw=0.5)

        # Add vertical lines to indicate the current frame
        self.vline_x = self.ax_x.axvline(self.curr_frame, color="k", linestyle="--")
        self.vline_y = self.ax_y.axvline(self.curr_frame, color="k", linestyle="--")

        # Configure legend (below video)
        custom_lines = [
            plt.Line2D([0], [0], color=self.colors[i], lw=4) for i in range(self.num_animals)
        ]
        self.ax_legend = self.fig.add_axes([0.15, 0.15, 0.3, 0.05], frame_on=False)
        self.ax_legend.axis("off")
        self.ax_legend.legend(
            custom_lines,
            [f"Animal {i + 1}" for i in range(self.num_animals)],
            loc="center",
            ncol=self.num_animals,
            frameon=False,
        )

    def update_frame(self, frame_idx):
        # Update current frame
        self.curr_frame = int(frame_idx)

        # Update video frame
        frame_path = os.path.join(self.video_frames_path, self.frames[self.curr_frame])
        img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        self.im_video.set_data(img)

        # Update scatter points
        scatter_data = self.track_data[:, self.curr_frame, :]
        self.scat_points.set_offsets(scatter_data)
        self.scat_points.set_color(self.colors)

        # Update vertical lines
        self.vline_x.set_xdata(self.curr_frame)
        self.vline_y.set_xdata(self.curr_frame)

        self.fig.canvas.draw_idle()

    def update_dot_size(self, size):
        # Update scatter point size
        self.point_size = int(size)
        self.scat_points.set_sizes([self.point_size] * len(self.scat_points.get_offsets()))
        self.fig.canvas.draw_idle()
    def show(self):
        plt.show()

# Usage example
if __name__ == "__main__":
    video_frames_path = "E:/01-LYX/new-research/udmt_project/newwww-2025-01-13/tmp/5-mice-1min/extracted-images"
    track_path = "E:/01-LYX/new-research/udmt_project/newwww-2025-01-13/tracking-results/5-mice-1min/5-mice-1min-whole-filter5.npy"
    viz = TrackletVisualizer(video_frames_path, track_path)
    viz.show()
