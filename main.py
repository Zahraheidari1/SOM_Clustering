from bokeh.layouts import row
from functions import run_multiple
from bokeh.plotting import curdoc, figure, show
from bokeh.models import FileInput, Button, Slider, ColumnDataSource, HoverTool,PreText,RangeSlider
from bokeh.layouts import column
from bokeh.palettes import inferno


alldatasource = ColumnDataSource(data=dict(x=[], y=[], labels=[], colors=[], transformed_labels=[]))
source = ColumnDataSource(data=dict(x=[], y=[], labels=[], colors=[], transformed_labels=[]))

sourceSilhouette = ColumnDataSource(data=dict(Cluster=[], Score=[]))
sourceCalinski = ColumnDataSource(data=dict(Cluster=[], Score=[]))
sourceDavies = ColumnDataSource(data=dict(Cluster=[], Score=[]))
sourceQuantization = ColumnDataSource(data=dict(Epoch=[], Error=[]))
sourceTopographic = ColumnDataSource(data=dict(Epoch=[], Error=[]))

point_colors = []

labelscluster = {}
testdata = []

def execute_clustering():
    global point_colors,labelscluster,testdata
    mincluster = rangeslider.value[0]
    maxcluster = rangeslider.value[1]

    testdata,labelscluster = run_multiple(
        mincluster,maxcluster,input_file.filename,
        sourceSilhouette,sourceCalinski,sourceDavies)

    slidertc.start = mincluster
    slidertc.end = maxcluster

    slider.end = mincluster
    slider.value = 0
    point_colors = [inferno(mincluster)[i-1] for i in labelscluster[mincluster]]
    # Create labels for each point by combining point number and cluster number
    labels = [i for i, _ in enumerate(labelscluster[mincluster], start=1)]
    alldatasource.data = dict(
        x=testdata[:, 0],
        y=testdata[:, 1],
        labels=labels,
        colors=point_colors,
        transformed_labels=labelscluster[mincluster]
    )
    source.data = dict(
        x=testdata[:, 0],
        y=testdata[:, 1],
        labels=labels,
        colors=point_colors,
        transformed_labels=labelscluster[mincluster]
    )

# Create a file input control to select the input file
input_file = FileInput(accept=".xlsx")

# Create a button to trigger the execution
execute_button = Button(label="Execute", button_type="success")
execute_button.on_click(execute_clustering)


slidertc = Slider(start=2, end=100, step=1, value=2, title="Total Cluster")

slider = Slider(start=0, end=1, step=1, value=0, title="Cluster")

rangeslider = RangeSlider(start=2, end=500, step=1, value=(2,100), title="Range cluster")

# Create a figure for the dot chart
p = figure(width=800, height=800, title="Dot Chart")
points = p.circle('x', 'y', size=5, fill_color='colors', source=source)

ptxt = PreText(text="", width=400, height=75)

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

def update_slidertc(attr, old, new):
    selected_cluster = new

    slider.end = selected_cluster
    slider.value = 0
    point_colors = [inferno(selected_cluster)[i-1] for i in labelscluster[selected_cluster]]
    # Create labels for each point by combining point number and cluster number
    labels = [i for i, _ in enumerate(labelscluster[selected_cluster], start=1)]
    alldatasource.data = dict(
        x=testdata[:, 0],
        y=testdata[:, 1],
        labels=labels,
        colors=point_colors,
        transformed_labels=labelscluster[selected_cluster]
    )
    source.data = dict(
        x=testdata[:, 0],
        y=testdata[:, 1],
        labels=labels,
        colors=point_colors,
        transformed_labels=labelscluster[selected_cluster]
    )
# Generate Bokeh plots for NMI, ARI, and Silhouette
p_silhouette = figure(title='Silhouette', x_axis_label='Cluster', y_axis_label='Score',plot_width=400, plot_height=400)
p_silhouette.line('Cluster', 'Score', source=sourceSilhouette, color="blue")
# Add hover tool to display point and cluster information
hover2 = HoverTool(tooltips=[("Cluster", "@Cluster"), ("Score", "@Score")])
p_silhouette.add_tools(hover2)

p_calinski = figure(title='Calinski', x_axis_label='Cluster', y_axis_label='Score',plot_width=400, plot_height=400)
p_calinski.line('Cluster', 'Score', source=sourceCalinski, color="blue")
p_calinski.add_tools(hover2)

p_davies = figure(title='Davies', x_axis_label='Cluster', y_axis_label='Score',plot_width=400, plot_height=400)
p_davies.line('Cluster', 'Score', source=sourceDavies, color="blue")
p_davies.add_tools(hover2)

# Add callback to update the plot based on the slider value
slider.on_change('value', update_slider)

slidertc.on_change('value', update_slidertc)

# Create figures for quantization error and topographic error
p_quantization = figure(title='Quantization Error', x_axis_label='Epoch', y_axis_label='Error',
                        plot_width=400, plot_height=400)
p_quantization.line('Epoch', 'Error', source=sourceQuantization, color="green")
p_quantization.add_tools(hover2)

p_topographic = figure(title='Topographic Error', x_axis_label='Epoch', y_axis_label='Error',
                       plot_width=400, plot_height=400)
p_topographic.line('Epoch', 'Error', source=sourceTopographic, color="red")
p_topographic.add_tools(hover2)


# Define the layout
layout = column(
    rangeslider,
    row(p_silhouette, p_calinski, p_davies),
    row(slidertc,
    slider),
    row(p,ptxt),
    input_file,
    execute_button,
    row(p_quantization, p_topographic)
)

# Add layout to the current document (Bokeh server)
curdoc().add_root(layout)

# Show the Bokeh server
show(curdoc())


