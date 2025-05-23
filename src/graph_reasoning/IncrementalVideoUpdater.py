import cv2
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def figure_to_image(fig: plt.Figure) -> np.ndarray:
    """
    Convert a fully rendered Matplotlib figure to an RGB image (numpy array).
    Ensures layout is tight and rendering is flushed.
    """
    fig.tight_layout()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    # Use Agg canvas bound directly to the figure
    canvas = FigureCanvas(fig)
    canvas.draw()

    width, height = canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
    return image

class IncrementalVideoUpdater:
    def __init__(self, output_filename='output.mp4', fps=10, logger=None, segment_duration=300):
        self.output_filename = output_filename
        self.fps = fps
        self.interval = 1 / fps
        self.logger = logger
        self.segment_duration = segment_duration  # Time in seconds per segment

        self.video_writer = None
        self.frame_width = None
        self.frame_height = None
        self.current_frame = None

        self._running = False
        self._thread = None
        self.segment_start_time = None
        self.segment_index = 0

    def _get_segment_filename(self):
        """
        Generate a unique filename for each video segment.
        """
        if self.segment_index == 0:
            return self.output_filename
        base, ext = self.output_filename.rsplit('.', 1)
        return f"{base}_part{self.segment_index}.{ext}"

    def init_writer(self, frame: np.ndarray):
        """
        Initialize the VideoWriter with frame dimensions.
        """
        self.frame_height, self.frame_width, _ = frame.shape
        filename = self._get_segment_filename()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filename, fourcc, self.fps, (self.frame_width, self.frame_height))
        self.segment_start_time = time.time()
        if self.logger:
            self.logger.info(f"VideoWriter initialized: {filename} ({self.frame_width}x{self.frame_height})")

    def _rotate_video_segment(self):
        """
        Close current segment and start a new one.
        """
        if self.video_writer:
            self.video_writer.release()
        self.segment_index += 1
        filename = self._get_segment_filename()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filename, fourcc, self.fps, (self.frame_width, self.frame_height))
        self.segment_start_time = time.time()
        if self.logger:
            self.logger.info(f"Started new video segment: {filename}")

    def _write_frames_sync(self):
        frame_count = 0
        if self.logger:
            self.logger.info("Started video writing loop.")
        while self._running:
            if self.current_frame is not None and self.video_writer is not None:
                self.video_writer.write(self.current_frame)
                frame_count += 1
                if self.logger:
                    self.logger.info(f"Frame {frame_count} written.")

                # Check if we need to start a new segment
                if time.time() - self.segment_start_time >= self.segment_duration:
                    self._rotate_video_segment()

            time.sleep(self.interval)
        if self.logger:
            self.logger.info("Stopped video writing loop.")

    def start(self):
        """
        Start the background video writing thread.
        """
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._write_frames_sync, daemon=True)
            self._thread.start()
            if self.logger:
                self.logger.info("Video writer thread started.")

    def update_figure(self, fig: plt.Figure):
        """
        Capture and convert a figure into a video frame.
        """
        image = figure_to_image(fig)
        if self.video_writer is None:
            self.init_writer(image)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.current_frame = image_bgr
        if self.logger:
            self.logger.info("Frame updated from figure.")

    def stop(self):
        """
        Stop the writer and finalize the video file.
        """
        self._running = False
        if self._thread:
            self._thread.join()
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        if self.logger:
            self.logger.info("Video writing stopped and finalized.")
