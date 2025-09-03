import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import cv2
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PySide6.QtWidgets import QMessageBox

plt.rcParams['figure.max_open_warning'] = 0

# Set default font settings with fallback
import matplotlib.font_manager as fm

# Try to find a suitable font family
font_families = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'sans-serif']
font_found = False

for font_family in font_families:
    try:
        # Check if font is available
        fm.findfont(font_family, fallback_to_default=False)
        plt.rcParams['font.family'] = font_family
        font_found = True
        break
    except:
        continue

# If no specific font found, use default sans-serif
if not font_found:
    plt.rcParams['font.family'] = 'sans-serif'

# Store the selected font family for use in other parts of the code
selected_font_family = plt.rcParams['font.family']

plt.rcParams['font.size'] = 10         # Base font size
plt.rcParams['axes.titlesize'] = 12    # Title font size
plt.rcParams['axes.labelsize'] = 10    # Axis label font size
plt.rcParams['xtick.labelsize'] = 9    # X-axis tick label font size
plt.rcParams['ytick.labelsize'] = 9    # Y-axis tick label font size

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind('<Enter>', self.enter)
        self.widget.bind('<Leave>', self.leave)

    def enter(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip, text=self.text, justify=tk.LEFT,
                        background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                        font=("tahoma", "8", "normal"))
        label.pack()

    def leave(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class TrackletVisualizer:
    def __init__(self, video_frames_path, track_path, time_points=None):
        # Remove Tkinter root window creation to avoid conflicts with Qt
        # self.root = tk.Tk()
        # self.root.withdraw()  # Hide the root window

        # Store time points for marking
        self.time_points = time_points if time_points is not None else []
        
        # Define marker color for time points
        self.marker_color = 'purple'  # Use consistent purple color for all time points
        
        # Load video frames (only .jpg files)
        self.video_frames_path = video_frames_path
        self.frames = sorted(
            [f for f in os.listdir(video_frames_path) if f.endswith(".jpg")]
        )

        self.nframes = len(self.frames)
        # Initialize start and end frame indices
        self.start_frame = 0
        self.end_frame = self.nframes - 1

        # Load track data (shape: [num_animals, num_frames, 2])
        self.track_data = np.load(track_path)
        self.num_animals, self.nframes_from_data, _ = self.track_data.shape

        # Store original track path
        self.track_path = track_path

        # Initialize selected animals for swapping
        self.selected_animals = []

        if self.nframes != self.nframes_from_data:
            # warning_message = (
            #     f"Warning: The number of video frames ({self.nframes}) does not match the number of frames in the track data ({self.nframes_from_data}).\n"
            #     "The video frames will be truncated to match the track data."
            # )
            # print(warning_message)
            self.frames = self.frames[:self.nframes_from_data]
            self.nframes = len(self.frames)

        # Colors for each animal
        self.colors = plt.cm.get_cmap('viridis', self.num_animals).colors
        # Highlight color for selected animals
        self.highlight_color = 'yellow'

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
        self.ax_slider = self.fig.add_axes([0.1, 0.1, 0.65, 0.03], facecolor="lightgray")
        self.slider = Slider(self.ax_slider, "Frame", 0, self.nframes - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.update_frame)

        # Add start and end frame indicators on the slider
        self.start_line = self.ax_slider.axvline(self.start_frame, color='g', linestyle='-', linewidth=2)
        self.end_line = self.ax_slider.axvline(self.end_frame, color='r', linestyle='-', linewidth=2)

        # Add time point markers on the slider
        self.time_markers = []
        for i, t in enumerate(self.time_points):
            if 0 <= t < self.nframes:
                marker = self.ax_slider.axvline(t, color=self.marker_color, linestyle='--', linewidth=1.5, alpha=0.7)
                self.time_markers.append(marker)

        # Add time points legend next to the slider
        self.ax_time_legend = self.fig.add_axes([0.8, 0.1, 0.2, 0.03], frame_on=False)
        self.ax_time_legend.axis("off")
        time_line = plt.Line2D([0], [0], color=self.marker_color, lw=4, linestyle='--')
        self.ax_time_legend.legend(
            [time_line],
            ["Low confidence frames"],
            loc="center left",
            frameon=False,
            prop={'family': selected_font_family, 'size': 9},
            handletextpad=0.5
        )

        # Add click event handler for the slider
        self.fig.canvas.mpl_connect('button_press_event', self.on_slider_click)

        self.ax_slider_size = self.fig.add_axes([0.1, 0.05, 0.5, 0.03], facecolor="orange")
        self.slider_size = Slider(self.ax_slider_size, "Dot Size", 10, 200, valinit=self.point_size, valstep=10)
        self.slider_size.on_changed(self.update_dot_size)

        # Add swap button
        self.ax_swap = self.fig.add_axes([0.65, 0.05, 0.15, 0.03])
        self.swap_button = Button(self.ax_swap, 'Swap Trajectories', color='lightblue', hovercolor='lightgreen')
        self.swap_button.on_clicked(self.swap_trajectories)
        
        # Add tooltip for swap button
        swap_tooltip_text = """How to swap trajectories:
1. Select start frame: Hold Shift + Click on slider
2. Select end frame: Hold Ctrl + Click on slider
3. Select animals: Click on animal legends (max 2)
4. Click Swap button to exchange trajectories"""
        
        # Create annotation for tooltip
        self.tooltip = self.ax_swap.annotate(swap_tooltip_text,
            xy=(0.5, 0.5), xytext=(0.5, 1.5),
            xycoords='axes fraction', textcoords='axes fraction',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray'),
            ha='center', va='bottom',
            visible=False,
            fontsize=9,
            fontfamily=selected_font_family)
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # Add save button
        self.ax_save = self.fig.add_axes([0.83, 0.05, 0.07, 0.03])
        self.save_button = Button(self.ax_save, 'Save', color='lightblue', hovercolor='lightgreen')
        self.save_button.on_clicked(self.save_trajectories)

        # Plot data placeholders
        self.im_video = None
        self.scat_points = None
        self.lines_x = []
        self.lines_y = []
        self.vline_x = None
        self.vline_y = None
        self.vline_start_x = None
        self.vline_start_y = None
        self.vline_end_x = None
        self.vline_end_y = None
        self.legend_lines = []  # Store legend lines for selection

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

        # Add scatter points for the current frame
        self.scat_points = self.ax_video.scatter([], [], s=self.point_size, edgecolor="white", lw=0.5)

        # Add shaded background for selected region
        self.shaded_region_x = self.ax_x.axvspan(self.start_frame, self.end_frame, color='gray', alpha=0.2)
        self.shaded_region_y = self.ax_y.axvspan(self.start_frame, self.end_frame, color='gray', alpha=0.2)

        # Add vertical lines to indicate the current frame (black dashed)
        self.vline_x = self.ax_x.axvline(self.curr_frame, color="k", linestyle="--")
        self.vline_y = self.ax_y.axvline(self.curr_frame, color="k", linestyle="--")

        # Add vertical lines to indicate start frame (green solid)
        self.vline_start_x = self.ax_x.axvline(self.start_frame, color="g", linestyle="-")
        self.vline_start_y = self.ax_y.axvline(self.start_frame, color="g", linestyle="-")

        # Add vertical lines to indicate end frame (red solid)
        self.vline_end_x = self.ax_x.axvline(self.end_frame, color="r", linestyle="-")
        self.vline_end_y = self.ax_y.axvline(self.end_frame, color="r", linestyle="-")

        # Add legend for frame indicators
        legend_elements = [
            plt.Line2D([0], [0], color='k', linestyle='--', label='Current Frame'),
            plt.Line2D([0], [0], color='g', linestyle='-', label='Start Frame (Shift + Click)'),
            plt.Line2D([0], [0], color='r', linestyle='-', label='End Frame (Ctrl + Click)'),
            plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.2, label='Selected Region')
        ]
        
        self.ax_x.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.8, 
                        fontsize='small', prop={'family': selected_font_family, 'size': 9})

        # Configure legend for animal colors (below video)
        self.ax_legend = self.fig.add_axes([0.15, 0.15, 0.3, 0.05], frame_on=False)
        self.ax_legend.axis("off")
        
        # Create clickable legend lines
        self.legend_lines = []
        for i in range(self.num_animals):
            line = plt.Line2D([0], [0], color=self.colors[i], lw=4, picker=True, pickradius=10)
            self.legend_lines.append(line)
        
        self.legend = self.ax_legend.legend(
            self.legend_lines,
            [f"Animal {i + 1}" for i in range(self.num_animals)],
            loc="center",
            ncol=self.num_animals,
            frameon=False,
            prop={'family': selected_font_family, 'size': 9},  # Font size
            handletextpad=0.5,  # Reduce spacing between line and text
            columnspacing=1.0    # Reduce spacing between columns
        )
        
        # Store original text colors
        self.original_text_colors = [text.get_color() for text in self.legend.get_texts()]
        
        # Make legend lines clickable
        for legend_line in self.legend.legendHandles:
            legend_line.set_picker(True)
            legend_line.set_pickradius(10)
        
        # Connect the pick event
        self.fig.canvas.mpl_connect('pick_event', self.on_legend_click)

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

    def on_slider_click(self, event):
        if event.inaxes == self.ax_slider:
            # Convert click position to frame index
            frame_idx = int(self.slider.valmin + (event.xdata - self.ax_slider.get_xlim()[0]) * 
                          (self.slider.valmax - self.slider.valmin) / 
                          (self.ax_slider.get_xlim()[1] - self.ax_slider.get_xlim()[0]))
            frame_idx = max(0, min(frame_idx, self.nframes - 1))

            # If shift key is pressed, set start frame
            if event.key == 'shift':
                if frame_idx <= self.end_frame:
                    self.start_frame = frame_idx
                    self.start_line.set_xdata(self.start_frame)
                    self.vline_start_x.set_xdata(self.start_frame)
                    self.vline_start_y.set_xdata(self.start_frame)
                    # Update shaded region
                    self.shaded_region_x.set_xy([[self.start_frame, 0], [self.start_frame, 1],
                                               [self.end_frame, 1], [self.end_frame, 0]])
                    self.shaded_region_y.set_xy([[self.start_frame, 0], [self.start_frame, 1],
                                               [self.end_frame, 1], [self.end_frame, 0]])
            # If control key is pressed, set end frame
            elif event.key == 'control':
                if frame_idx >= self.start_frame:
                    self.end_frame = frame_idx
                    self.end_line.set_xdata(self.end_frame)
                    self.vline_end_x.set_xdata(self.end_frame)
                    self.vline_end_y.set_xdata(self.end_frame)
                    # Update shaded region
                    self.shaded_region_x.set_xy([[self.start_frame, 0], [self.start_frame, 1],
                                               [self.end_frame, 1], [self.end_frame, 0]])
                    self.shaded_region_y.set_xy([[self.start_frame, 0], [self.start_frame, 1],
                                               [self.end_frame, 1], [self.end_frame, 0]])
            # Otherwise, update current frame
            else:
                self.slider.set_val(frame_idx)

            self.fig.canvas.draw_idle()

    def update_dot_size(self, size):
        # Update scatter point size
        self.point_size = int(size)
        self.scat_points.set_sizes([self.point_size] * len(self.scat_points.get_offsets()))
        self.fig.canvas.draw_idle()

    def on_legend_click(self, event):
        if event.artist in self.legend_lines or event.artist in self.legend.legendHandles:
            # Get the index of the clicked line
            if event.artist in self.legend_lines:
                animal_idx = self.legend_lines.index(event.artist)
            else:
                animal_idx = self.legend.legendHandles.index(event.artist)
            
            if animal_idx in self.selected_animals:
                # Deselect
                self.selected_animals.remove(animal_idx)
                self.legend.get_texts()[animal_idx].set_color(self.original_text_colors[animal_idx])
            else:
                # Select
                if len(self.selected_animals) < 2:
                    self.selected_animals.append(animal_idx)
                    self.legend.get_texts()[animal_idx].set_color('red')
            
            self.fig.canvas.draw_idle()

    def swap_trajectories(self, event):
        if len(self.selected_animals) != 2:
            # Use Qt message box instead of tkinter messagebox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please select exactly two animals to swap")
            msg.setWindowTitle("Warning")
            msg.exec_()
            return
        
        if self.start_frame >= self.end_frame:
            # Use Qt message box instead of tkinter messagebox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please select valid start and end frames")
            msg.setWindowTitle("Warning")
            msg.exec_()
            return

        # Get the two animal indices
        animal1, animal2 = self.selected_animals

        # Swap trajectories in the selected range
        temp_data = self.track_data[animal1, self.start_frame:self.end_frame+1].copy()
        self.track_data[animal1, self.start_frame:self.end_frame+1] = self.track_data[animal2, self.start_frame:self.end_frame+1]
        self.track_data[animal2, self.start_frame:self.end_frame+1] = temp_data

        # Update the plots
        self.lines_x[animal1].set_ydata(self.track_data[animal1, :, 0])
        self.lines_y[animal1].set_ydata(self.track_data[animal1, :, 1])
        self.lines_x[animal2].set_ydata(self.track_data[animal2, :, 0])
        self.lines_y[animal2].set_ydata(self.track_data[animal2, :, 1])

        # Update scatter points
        scatter_data = self.track_data[:, self.curr_frame, :]
        self.scat_points.set_offsets(scatter_data)

        # Reset selection and text colors
        for idx in self.selected_animals:
            self.legend.get_texts()[idx].set_color(self.original_text_colors[idx])
        self.selected_animals = []

        self.fig.canvas.draw_idle()

    def save_trajectories(self, event):
        # Generate new filename
        original_path = self.track_path
        save_path = original_path.replace('.npy', '-manual-correction.npy')
        
        # Save the corrected trajectories
        np.save(save_path, self.track_data)
        # Use Qt message box instead of tkinter messagebox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"Corrected trajectories saved to:\n{save_path}")
        msg.setWindowTitle("Success")
        msg.exec_()

    def on_mouse_move(self, event):
        if event.inaxes == self.ax_swap:
            self.tooltip.set_visible(True)
        else:
            self.tooltip.set_visible(False)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()
        # Remove root.destroy() since we no longer create a Tkinter root window
        # self.root.destroy()  # Clean up the root window when closing

# Usage example
if __name__ == "__main__":
    video_frames_path = "E:/01-LYX/new-research/udmt_project/7-mice-2025-01-22/tmp/7-mice-1min-v2/extracted-images"
    track_path = "E:/01-LYX/new-research/udmt_project/7-mice-2025-01-22/tracking-results/7-mice-1min-v2/7-mice-1min-v2-whole-filter5.npy"
    time_points = [2, 250, 900]  # Add time points for marking
    viz = TrackletVisualizer(video_frames_path, track_path, time_points=time_points)
    viz.show()
