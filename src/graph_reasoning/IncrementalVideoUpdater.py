import cv2
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def figure_to_image(fig: plt.Figure) -> np.ndarray:
    """
    Convert a Matplotlib figure to an RGB image (numpy array).
    """
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    return image

class IncrementalVideoUpdater:
    def __init__(self, output_filename='output.mp4', fps=10, logger=None, flush_interval=300):
        self.output_filename = output_filename
        self.fps = fps
        self.interval = 1 / fps
        self.logger = logger
        self.flush_interval = flush_interval  # Time in seconds between flushing the video

        self.video_writer = None
        self.frame_width = None
        self.frame_height = None
        self.current_frame = None

        self._running = False
        self._thread = None
        self.segment_start_time = None  # Track the start time of the video file

    def init_writer(self, frame: np.ndarray):
        """
        Initialize the VideoWriter using the dimensions of the provided frame.
        """
        self.logger.info(f"VideoWriter initialized with resolution: {self.frame_width}x{self.frame_height}")

        self.frame_height, self.frame_width, _ = frame.shape
        # Try using a codec like XVID for compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec (mp4v)
        self.video_writer = cv2.VideoWriter(
            self.output_filename, fourcc, self.fps, (self.frame_width, self.frame_height)
        )
        self.segment_start_time = time.time()  # Track the start time
        self.logger.info(f"VideoWriter initialized with resolution: {self.frame_width}x{self.frame_height}")

    def _write_frames_sync(self):
        frame_count = 0
        if self.logger:
            self.logger.info("Entering frame writing loop.")
        while self._running:
            if self.current_frame is not None and self.video_writer is not None:
                self.video_writer.write(self.current_frame)
                frame_count += 1
                if self.logger:
                    self.logger.info(f"Frame {frame_count} written, shape: {self.current_frame.shape}")

                # Periodically flush the video writer
                current_time = time.time()
                if current_time - self.segment_start_time > self.flush_interval:
                    self.flush_video()
                    self.segment_start_time = current_time  # Reset the start time

            time.sleep(self.interval)
        if self.logger:
            self.logger.info("Exiting frame writing loop.")

    def start(self):
        """
        Start the dedicated thread for video frame writing.
        """
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._write_frames_sync, daemon=True)
            self._thread.start()
            self.logger.info("Video writing started.")

    def update_figure(self, fig: plt.Figure):
        """
        Update the current frame using a new matplotlib figure.
        The figure is converted to an image and then used as the next video frame.
        """
        image = figure_to_image(fig)
        if self.video_writer is None:
            print("Initializing VideoWriter...")
            self.init_writer(image)
        # Convert from RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.video_writer is None:
            self.init_writer(image)
        self.current_frame = image
        if self.logger:
            self.logger.info(f"Frame updated from matplotlib figure with shape {image.shape}")
        self.logger.info("Frame updated from matplotlib figure.")

    def flush_video(self):
        """Flush the video writer (ensure data is written periodically)."""
        if self.video_writer:
            self.video_writer.release()  # Release the writer
            self.video_writer = cv2.VideoWriter(self.output_filename, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.frame_width, self.frame_height))
            self.logger.info(f"Flushed video and reopened {self.output_filename}")

    def stop(self):
        """
        Stop the frame writing loop and finalize the video file.
        """
        self._running = False
        if self._thread:
            self._thread.join()
        if self.video_writer is not None:
            self.video_writer.release()
        if self.logger:
            self.logger.info("Video writing stopped and file finalized.")
        self.logger.info("Video writing stopped and file finalized.")
