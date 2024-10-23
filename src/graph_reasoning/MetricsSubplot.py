import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

class MetricsSubplot:
    def __init__(self, nrows=2, ncols=2, plot_names_map = {}, figsize=(10, 8)):
        """
        Initialize the MetricsSubplot with a grid of subplots.
        
        Parameters:
            nrows (int): Number of rows of subplots.
            ncols (int): Number of columns of subplots.
            figsize (tuple): Size of the figure.
        """
        self.nrows = nrows
        self.ncols = ncols
        self.plot_names_map = plot_names_map
        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        self.axes = self.axes.flatten()  # Flatten for easier indexing if 2D

    def update_plot_with_figure(self, name, fig):
        """
        Update a specific subplot using an existing plt.Figure.
        
        Parameters:
            index (int): Index of the subplot to update (0-based).
            fig (plt.Figure): A pre-existing matplotlib figure to display.
        """
        index = self.plot_names_map[name]
        if index >= len(self.axes):
            raise ValueError(f"Index {index} out of range for subplot grid size {len(self.axes)}.")
        
        ax = self.axes[index]
        ax.clear()  # Clear the current plot

        # Transfer elements of the pre-built figure to the target ax
        for child_ax in fig.get_axes():
            # Redraw lines, images, etc., onto the target ax.
            for line in child_ax.get_lines():
                ax.plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    label=line.get_label(),
                    color=line.get_color(),
                    linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth(),
                    alpha=line.get_alpha(),  # Preserve alpha value
                )
            for img in child_ax.get_images():
                ax.imshow(
                    img.get_array(),
                    extent=img.get_extent(),
                    aspect=img.get_aspect(),
                    cmap=img.get_cmap(),
                    alpha=img.get_alpha()  # Preserve alpha for images as well
                )
            
            # Copy titles and labels from the child axis
            ax.set_xlabel(child_ax.get_xlabel())
            ax.set_ylabel(child_ax.get_ylabel())
            ax.legend(loc='best')

        # Set the custom title if provided
        if name:
            ax.set_title(name)
        else:
            # Fallback to the title from the child axis if no custom title is provided
            ax.set_title(child_ax.get_title())

        # Close the original figure to prevent it from being shown separately
        plt.close(fig)
        self.fig.tight_layout()

    def show(self, block=False):
        """Display the updated figure."""
        plt.show(block=False)
        plt.pause(0.001)  # A small pause to allow the figure to render


    def save(self, filename):
        """
        Save the current state of the figure to a file.
        
        Parameters:
            filename (str): The name of the file to save the figure.
        """
        self.fig.savefig(filename)
