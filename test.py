import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, Div, Select
from bokeh.plotting import figure
from bokeh.server.server import Server
from tornado.ioloop import IOLoop
from bokeh.layouts import row, column
from bokeh.models import Select
from bokeh.plotting import figure, show

# Function to generate random line data (replace with your actual line data)
def generate_bokeh_plots(param_grid, nmi_values, ari_values, silhouette_values, best_map_size, best_num_epochs):
    # Generate Bokeh plots
    p_nmi = figure(title='NMI', x_axis_label='num_epochs', y_axis_label='NMI')
    p_nmi.line(param_grid['num_epochs'], nmi_values, color="blue", alpha=0.3)

    p_ari = figure(title='ARI', x_axis_label='num_epochs', y_axis_label='ARI')
    p_ari.line(param_grid['num_epochs'], ari_values, color="green", alpha=0.3)

    p_silhouette = figure(title='Silhouette', x_axis_label='num_epochs', y_axis_label='Silhouette')
    p_silhouette.line(param_grid['num_epochs'], silhouette_values, color="red", alpha=0.3)

    # Create select menus for map size and num_epochs
    map_size_select = Select(title="Map Size:", value=str(best_map_size),
                             options=[str(x) for x in param_grid['map_size']])
    num_epochs_select = Select(title="Num Epochs:", value=str(best_num_epochs),
                               options=[str(x) for x in param_grid['num_epochs']])

    def update_selections(attrname, old, new):
        selected_map_size = eval(map_size_select.value)
        selected_num_epochs = int(num_epochs_select.value)

        # Update the line graphs with the selected parameters
        p_nmi.line(param_grid['num_epochs'], nmi_values, color="blue", alpha=0.3)
        p_nmi.line([selected_num_epochs], [nmi_values[param_grid['num_epochs'].index(selected_num_epochs)]],
                   color="blue", alpha=1.0, line_width=2)

        p_ari.line(param_grid['num_epochs'], ari_values, color="green", alpha=0.3)
        p_ari.line([selected_num_epochs], [ari_values[param_grid['num_epochs'].index(selected_num_epochs)]],
                   color="green", alpha=1.0, line_width=2)

        p_silhouette.line(param_grid['num_epochs'], silhouette_values, color="red", alpha=0.3)
        p_silhouette.line([selected_num_epochs],
                          [silhouette_values[param_grid['num_epochs'].index(selected_num_epochs)]],
                          color="red", alpha=1.0, line_width=2)

    map_size_select.on_change('value', update_selections)
    num_epochs_select.on_change('value', update_selections)

    # Arrange the plots and select menus in a layout
    layout = row(column(map_size_select, num_epochs_select), p_nmi, p_ari, p_silhouette)

    # Show the layout
    show(layout)


# Example usage
generate_bokeh_plots(param_grid, nmi_values, ari_values, silhouette_values, best_map_size, best_num_epochs)


'''normalizer = hazm.Normalizer()
tokenizer = hazm.WordTokenizer()
stemmer = hazm.Stemmer()
docs = [normalizer.character_refinement(doc) for doc in docs if doc is not docs]
docs = [' '.join(tokenizer.tokenize(doc)) for doc in docs if doc is not docs]
docs=[stemmer.stem(doc) for doc in docs]
vectorizer = CountVectorizer(stop_words=hazm.stopwords_list())
text=['انهدام كامل مجتمع توليد سلاحهاي ميكروبي ']
print(docs)
doc_vectors = vectorizer.fit_transform(docs)


similarity_matrix = cosine_similarity(doc_vectors)


doc_index = 0
similar_docs = pd.DataFrame(similarity_matrix[doc_index], columns=['similarity'])
similar_docs['document'] = docs
similar_docs = similar_docs.sort_values(by='similarity', ascending=False)
print(similar_docs)'''
