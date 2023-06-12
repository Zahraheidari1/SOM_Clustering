import random
import pandas as pd
import numpy as np
from bokeh.layouts import row
from mini import generate_cluster_labels
from bokeh.plotting import curdoc, figure, show
from bokeh.models import FileInput, Button, Slider, LabelSet, ColumnDataSource, HoverTool
from SOM import SOM
from sklearn.decomposition import PCA
from bokeh.layouts import column
from bokeh.palettes import inferno
from matplotlib import pyplot as plt
from bokeh.models import LinearColorMapper
from bokeh.layouts import gridplot


num_clusters = 101
colors = inferno(num_clusters)

alldatasource = ColumnDataSource(data=dict(x=[], y=[], labels=[], colors=[], transformed_labels=[]))
source = ColumnDataSource(data=dict(x=[], y=[], labels=[], colors=[], transformed_labels=[]))
point_colors = []

def execute_clustering():
    global point_colors

    # Call the generate_cluster_labels function with the selected file
    test_docs_vector, cluster_labels, u_matrix = generate_cluster_labels(input_file.filename, int(slider.value))
    # Display a message indicating successful execution
    print("Function executed successfully.")

    transformed_labels = [(label[0] * 10 + label[1]) + 1 for label in cluster_labels]

    point_colors = [colors[label] for label in transformed_labels]

    # Create labels for each point by combining point number and cluster number
    labels = [i for i, cluster in enumerate(transformed_labels, start=1)]
    alldatasource.data = dict(x=test_docs_vector[:, 0],
                              y=test_docs_vector[:, 1],
                              labels=labels,
                              colors=point_colors,
                              transformed_labels=transformed_labels)
    source.data = dict(x=test_docs_vector[:, 0],
                       y=test_docs_vector[:, 1],
                       labels=labels,
                       colors=point_colors,
                       transformed_labels=transformed_labels)


# Create a file input control to select the input file
input_file = FileInput(accept=".xlsx")

# Create a button to trigger the execution
execute_button = Button(label="Execute", button_type="success")
execute_button.on_click(execute_clustering)

# Create a slider that is between 0 and 100
slider = Slider(start=0, end=100, step=1, value=0, title="Cluster")

# Create a figure for the dot chart
p = figure(width=600, height=400, title="Dot Chart")
points = p.circle('x', 'y', size=5, fill_color='colors', alpha=0.5, source=source)

# Add hover tool to display point and cluster information
hover = HoverTool(tooltips=[("Document number", "@labels"), ("Cluster number", "@transformed_labels")])
p.add_tools(hover)

def update_slider(attr, old, new):
    selected_cluster = new

    if selected_cluster == 0:
        # Show all data points when the selected cluster is 0
        filtered_data = dict(
            x=alldatasource.data['x'],
            y=alldatasource.data['y'],
            labels=alldatasource.data['labels'],
            colors=alldatasource.data['colors'],
            transformed_labels=alldatasource.data['transformed_labels']
        )
    else:
        # Filter the data based on the selected cluster
        filtered_data = dict(
            x=[alldatasource.data['x'][i] for i, label in enumerate(alldatasource.data['transformed_labels']) if label == selected_cluster],
            y=[alldatasource.data['y'][i] for i, label in enumerate(alldatasource.data['transformed_labels']) if label == selected_cluster],
            labels=[alldatasource.data['labels'][i] for i, label in enumerate(alldatasource.data['transformed_labels']) if label == selected_cluster],
            colors=[alldatasource.data['colors'][i] for i, label in enumerate(alldatasource.data['transformed_labels']) if label == selected_cluster],
            transformed_labels=[alldatasource.data['transformed_labels'][i] for i, label in enumerate(alldatasource.data['transformed_labels']) if label == selected_cluster]
        )

    # Update the data source for the plot
    source.data = filtered_data

    # Update the data source for the points glyph
    points.data_source.data = filtered_data

# Add callback to update the plot based on the slider value
slider.on_change('value', update_slider)

# Create the U-matrix figure
u_matrix_figure = figure(width=600, height=400, title="U-Matrix")
hitmap_figure =figure(width=400 ,height=400 ,title="hitmap")
# Create the layout
layout = column(slider, p, u_matrix_figure,hitmap_figure, input_file, execute_button)

# Add the layout to the document
curdoc().add_root(layout)
