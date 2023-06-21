from bokeh.layouts import row
from functions import run_multiple, run_multiple_compare
from bokeh.plotting import curdoc, figure, show
from bokeh.models import FileInput, Button, Slider, ColumnDataSource, HoverTool,PreText,RangeSlider
from bokeh.layouts import column
from bokeh.palettes import inferno


alldatasource = ColumnDataSource(data=dict(x=[], y=[], labels=[], colors=[], transformed_labels=[]))
source = ColumnDataSource(data=dict(x=[], y=[], labels=[], colors=[], transformed_labels=[]))

sourceSilhouette = ColumnDataSource(data=dict(Cluster=[], Score=[]))
sourceCalinski = ColumnDataSource(data=dict(Cluster=[], Score=[]))
sourceDunn = ColumnDataSource(data=dict(Cluster=[], Score=[]))
sourceTopographic = ColumnDataSource(data=dict(Cluster=[], Error=[]))
sourceQuantization = ColumnDataSource(data=dict(Cluster=[], Error=[]))

sourceCompare = [[ColumnDataSource(data=dict(Iteration=[], Score=[])) for _ in range(3)] for _ in range(4)]

testsdata = []
Allclusterlabel = {}

def execute_clustering():
    global Allclusterlabel,testsdata
    mincluster = rangeslider.value[0]
    maxcluster = rangeslider.value[1]

    testsdata,Allclusterlabel = run_multiple(mincluster,maxcluster,input_file.filename,sliderrep.value)
    sourceSilhouette.data
    slidertc.start = mincluster
    slidertc.end = maxcluster

    slider.end = mincluster
    slider.value = 0
    point_colors = [inferno(mincluster)[i-1] for i in Allclusterlabel[mincluster][1][0]]
    # Create labels for each point by combining point number and cluster number
    labels = [i for i, _ in enumerate(Allclusterlabel[mincluster][1][0], start=1)]
    alldatasource.data = dict(
        x=testsdata[0][:, 0],
        y=testsdata[0][:, 1],
        labels=labels,
        colors=point_colors,
        transformed_labels=Allclusterlabel[mincluster][1][0]
    )
    source.data = dict(
        x=testsdata[0][:, 0],
        y=testsdata[0][:, 1],
        labels=labels,
        colors=point_colors,
        transformed_labels=Allclusterlabel[mincluster][1][0]
    )
    sourceSilhouette.data = dict(
        Cluster=list(range(mincluster,maxcluster+1)),
        Score = [Allclusterlabel[i][0]["silhouette"] for i in range(mincluster,maxcluster+1)]
    )
    sourceCalinski.data = dict(
        Cluster=list(range(mincluster,maxcluster+1)),
        Score = [Allclusterlabel[i][0]["calinski"] for i in range(mincluster,maxcluster+1)]
    )
    sourceDunn.data = dict(
        Cluster=list(range(mincluster,maxcluster+1)),
        Score = [Allclusterlabel[i][0]["dunn"] for i in range(mincluster,maxcluster+1)]
    )
    sourceTopographic.data = dict(
        Cluster=list(range(mincluster,maxcluster+1)),
        Error = [Allclusterlabel[i][0]["topographic"] for i in range(mincluster,maxcluster+1)]
    )
    sourceQuantization.data = dict(
        Cluster=list(range(mincluster,maxcluster+1)),
        Error = [Allclusterlabel[i][0]["quantization"] for i in range(mincluster,maxcluster+1)]
    )

# Create a file input control to select the input file
input_file = FileInput(accept=".xlsx")

# Create a button to trigger the execution
execute_button = Button(label="Execute 1", button_type="success")
execute_button.on_click(execute_clustering)


slidertc = Slider(start=2, end=100, step=1, value=2, title="Total Cluster")

slidertc2 = Slider(start=1,end=10,step=1,value=1,title="Number")

sliderrep = Slider(start=1,end=100,step=1,value=10,title="Repeat")

slider = Slider(start=0, end=1, step=1, value=0, title="Cluster")

rangeslider = RangeSlider(start=2, end=500, step=1, value=(2,100), title="Range cluster")

# Create a figure for the dot chart
p = figure(width=800, height=800, title="Dot Chart")
points = p.circle('x', 'y', size=5, fill_color='colors', source=source)

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
    point_colors = [inferno(selected_cluster)[i-1] for i in Allclusterlabel[selected_cluster][1][slidertc2.value-1]]
    # Create labels for each point by combining point number and cluster number
    labels = [i for i, _ in enumerate(Allclusterlabel[selected_cluster][1][slidertc2.value-1], start=1)]
    alldatasource.data = dict(
        x=testsdata[slidertc2.value-1][:, 0],
        y=testsdata[slidertc2.value-1][:, 1],
        labels=labels,
        colors=point_colors,
        transformed_labels=Allclusterlabel[selected_cluster][1][slidertc2.value-1]
    )
    source.data = dict(
        x=testsdata[slidertc2.value-1][:, 0],
        y=testsdata[slidertc2.value-1][:, 1],
        labels=labels,
        colors=point_colors,
        transformed_labels=Allclusterlabel[selected_cluster][1][slidertc2.value-1]
    )

def update_slidertc2(attr, old, new):
    selected_cluster = slidertc.value
    selected_number = slidertc2.value
    slider.end = selected_cluster
    slider.value = 0
    point_colors = [inferno(selected_cluster)[i-1] for i in Allclusterlabel[selected_cluster][1][selected_number-1]]
    # Create labels for each point by combining point number and cluster number
    labels = [i for i, _ in enumerate(Allclusterlabel[selected_cluster][1][selected_number-1], start=1)]
    alldatasource.data = dict(
        x=testsdata[selected_number-1][:, 0],
        y=testsdata[selected_number-1][:, 1],
        labels=labels,
        colors=point_colors,
        transformed_labels=Allclusterlabel[selected_cluster][1][selected_number-1]
    )
    source.data = dict(
        x=testsdata[selected_number-1][:, 0],
        y=testsdata[selected_number-1][:, 1],
        labels=labels,
        colors=point_colors,
        transformed_labels=Allclusterlabel[selected_cluster][1][selected_number-1]
    )

def update_sliderrep(attr, old, new):
    slidertc2.end=new

# Generate Bokeh plots for NMI, ARI, and Silhouette
p_silhouette = figure(title='Silhouette', x_axis_label='Cluster', y_axis_label='Score',plot_width=400, plot_height=400)
p_silhouette.line('Cluster', 'Score', source=sourceSilhouette, color="blue")
# Add hover tool to display point and cluster information
hover2 = HoverTool(tooltips=[("Cluster", "@Cluster"), ("Score", "@Score")])
p_silhouette.add_tools(hover2)

p_calinski = figure(title='Calinski', x_axis_label='Cluster', y_axis_label='Score',plot_width=400, plot_height=400)
p_calinski.line('Cluster', 'Score', source=sourceCalinski, color="blue")
p_calinski.add_tools(hover2)

p_dunn = figure(title='Dunn', x_axis_label='Cluster', y_axis_label='Score',plot_width=400, plot_height=400)
p_dunn.line('Cluster', 'Score', source=sourceDunn, color="blue")
p_dunn.add_tools(hover2)

hover3 = HoverTool(tooltips=[("Cluster", "@Cluster"), ("Error", "@Error")])

p_topographic = figure(title='Topographic', x_axis_label='Cluster', y_axis_label='Error',plot_width=400, plot_height=400)
p_topographic.line('Cluster', 'Error', source=sourceTopographic, color="blue")
p_topographic.add_tools(hover3)

p_quantization = figure(title='Quantization', x_axis_label='Cluster', y_axis_label='Error',plot_width=400, plot_height=400)
p_quantization.line('Cluster', 'Error', source=sourceQuantization, color="blue")
p_quantization.add_tools(hover3)

# Add callback to update the plot based on the slider value
slider.on_change('value', update_slider)

slidertc.on_change('value', update_slidertc)

slidertc2.on_change('value', update_slidertc2)

sliderrep.on_change('value', update_sliderrep)

# for comparing 
slideritrate = Slider(start=1, end=100, step=1, value=1, title="Iteration count")

hover4 = HoverTool(tooltips=[("Iteration", "@Iteration"), ("Score", "@Score")])
p_silhouette_Comp = figure(title='Silhouette', x_axis_label='Iteration', y_axis_label='Score',plot_width=400, plot_height=400)
p_silhouette_Comp.line('Iteration', 'Score', source=sourceCompare[0][0], color="blue" , legend_label="SOM")
p_silhouette_Comp.line('Iteration', 'Score', source=sourceCompare[1][0], color="red" , legend_label="Kmeans")
p_silhouette_Comp.line('Iteration', 'Score', source=sourceCompare[2][0], color="purple" , legend_label="Agglomerative")
p_silhouette_Comp.line('Iteration', 'Score', source=sourceCompare[3][0], color="green" , legend_label="MiniSom")
p_silhouette_Comp.legend.click_policy="mute"
p_silhouette_Comp.add_tools(hover4)

p_calinski_Comp = figure(title='Calinski', x_axis_label='Iteration', y_axis_label='Score',plot_width=400, plot_height=400)
p_calinski_Comp.line('Iteration', 'Score', source=sourceCompare[0][1], color="blue" , legend_label="SOM")
p_calinski_Comp.line('Iteration', 'Score', source=sourceCompare[1][1], color="red" , legend_label="Kmeans")
p_calinski_Comp.line('Iteration', 'Score', source=sourceCompare[2][1], color="purple" , legend_label="Agglomerative")
p_calinski_Comp.line('Iteration', 'Score', source=sourceCompare[3][1], color="green" , legend_label="MiniSom")
p_calinski_Comp.legend.click_policy="mute"
p_calinski_Comp.add_tools(hover4)

p_dunn_Comp = figure(title='Dunn', x_axis_label='Iteration', y_axis_label='Score',plot_width=400, plot_height=400)
p_dunn_Comp.line('Iteration', 'Score', source=sourceCompare[0][2], color="blue" , legend_label="SOM")
p_dunn_Comp.line('Iteration', 'Score', source=sourceCompare[1][2], color="red" , legend_label="Kmeans")
p_dunn_Comp.line('Iteration', 'Score', source=sourceCompare[2][2], color="purple" , legend_label="Agglomerative")
p_dunn_Comp.line('Iteration', 'Score', source=sourceCompare[3][2], color="green" , legend_label="MiniSom")
p_dunn_Comp.legend.click_policy="mute"
p_dunn_Comp.add_tools(hover4)

def execute_clustering_comp():
    datas = run_multiple_compare(slidertc.value,slideritrate.value,input_file.filename,sliderrep.value)
    sourceCompare[0][0].data = dict(
        Iteration = list(range(1,slideritrate.value+1)),
        Score = [i[1]["SOM"]["silhouette"] for i in list(sorted(datas.items()))]
    )
    sourceCompare[0][1].data = dict(
        Iteration = list(range(1,slideritrate.value+1)),
        Score = [i[1]["SOM"]["calinski"] for i in list(sorted(datas.items()))]
    )
    sourceCompare[0][2].data = dict(
        Iteration = list(range(1,slideritrate.value+1)),
        Score = [i[1]["SOM"]["dunn"] for i in list(sorted(datas.items()))]
    )

    sourceCompare[1][0].data = dict(
        Iteration = list(range(1,slideritrate.value+1)),
        Score = [i[1]["Kmeans"]["silhouette"] for i in list(sorted(datas.items()))]
    )
    sourceCompare[1][1].data = dict(
        Iteration = list(range(1,slideritrate.value+1)),
        Score = [i[1]["Kmeans"]["calinski"] for i in list(sorted(datas.items()))]
    )
    sourceCompare[1][2].data = dict(
        Iteration = list(range(1,slideritrate.value+1)),
        Score = [i[1]["Kmeans"]["dunn"] for i in list(sorted(datas.items()))]
    )

    sourceCompare[2][0].data = dict(
        Iteration = list(range(1,slideritrate.value+1)),
        Score = [i[1]["Agglomerative"]["silhouette"] for i in list(sorted(datas.items()))]
    )
    sourceCompare[2][1].data = dict(
        Iteration = list(range(1,slideritrate.value+1)),
        Score = [i[1]["Agglomerative"]["calinski"] for i in list(sorted(datas.items()))]
    )
    sourceCompare[2][2].data = dict(
        Iteration = list(range(1,slideritrate.value+1)),
        Score = [i[1]["Agglomerative"]["dunn"] for i in list(sorted(datas.items()))]
    )

    sourceCompare[3][0].data = dict(
        Iteration = list(range(1,slideritrate.value+1)),
        Score = [i[1]["MiniSom"]["silhouette"] for i in list(sorted(datas.items()))]
    )
    sourceCompare[3][1].data = dict(
        Iteration = list(range(1,slideritrate.value+1)),
        Score = [i[1]["MiniSom"]["calinski"] for i in list(sorted(datas.items()))]
    )
    sourceCompare[3][2].data = dict(
        Iteration = list(range(1,slideritrate.value+1)),
        Score = [i[1]["MiniSom"]["dunn"] for i in list(sorted(datas.items()))]
    )
        

# Create a button to trigger the execution
execute_button_comp = Button(label="Execute 2", button_type="success")
execute_button_comp.on_click(execute_clustering_comp)

# Define the layout
layout = column(
    row(rangeslider,sliderrep),
    row(p_silhouette, p_calinski, p_dunn),
    row(p_topographic, p_quantization),
    row(slidertc,slidertc2),
    slider,
    p,
    input_file,
    execute_button,
    slideritrate,
    row(p_silhouette_Comp, p_calinski_Comp, p_dunn_Comp),
    execute_button_comp
)

# Add layout to the current document (Bokeh server)
curdoc().add_root(layout)

# Show the Bokeh server
show(curdoc())