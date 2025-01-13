import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

class MetricsSubplot:
    def __init__(self, name, nrows=2, ncols=2, plot_names_map = {}, figsize=(13, 13)):
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
        self.fig.suptitle(name)
        self.fig.canvas.manager.set_window_title("Results")
        self.axes = self.axes.flatten()

        matplotlib.use('Agg')  # Use the non-interactive Agg backend

    def update_plot_with_figure(self, name, fig, square_it = False):
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
                    marker=line.get_marker(),  # Preserve marker style
                    markersize=line.get_markersize(),  # Preserve marker size
                    markerfacecolor=line.get_markerfacecolor(),  # Preserve marker face color
                    markeredgecolor=line.get_markeredgecolor(),  # Preserve marker edge color
                )
            for img in child_ax.get_images():
                ax.imshow(
                    img.get_array(),
                    extent=img.get_extent(),
                    aspect=img.get_aspect(),
                    cmap=img.get_cmap(),
                    alpha=img.get_alpha()  # Preserve alpha for images as well
                )
            
            _, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend()
            ax.set_xlabel(child_ax.get_xlabel())
            ax.set_ylabel(child_ax.get_ylabel())
            if ax.get_legend() is not None:
                ax.legend(loc='best', prop={'size': 6})

        # Set the custom title if provided
        if name:
            ax.set_title(name)
        else:
            # Fallback to the title from the child axis if no custom title is provided
            ax.set_title(child_ax.get_title())

        plt.close(fig)
        if square_it:
            ax.set_aspect('equal', adjustable='box') 
        # self.fig.tight_layout()

    def save(self, filename):
        """
        Save the current state of the figure to a file.
        
        Parameters:
            filename (str): The name of the file to save the figure.
        """
        self.fig.savefig(filename)

    def close(self):
        plt.close(self.fig)
