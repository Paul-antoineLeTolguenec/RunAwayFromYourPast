import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np

# Initialiser l'application Dash
app = dash.Dash(__name__)

# Définir la mise en page de l'application
app.layout = html.Div([
    dcc.Graph(id='graph-3d'),
    html.Button('Ajouter un point', id='btn-add-point', n_clicks=0)
])

# Générer des points 3D aléatoires
N = 100
x = np.random.rand(N)
y = np.random.rand(N)
z = np.random.rand(N)

# Callback pour mettre à jour la figure
@app.callback(
    Output('graph-3d', 'figure'),
    [Input('btn-add-point', 'n_clicks')]
)
def update_figure(n_clicks):
    # Déterminer le nombre de points à afficher
    num_points = min(n_clicks, N)

    # Créer la figure avec les points à afficher
    fig = go.Figure(data=[go.Scatter3d(
        x=x[:num_points],
        y=y[:num_points],
        z=z[:num_points],
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.8)
    )])

    # Configurer le layout de la figure
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    return fig

# Exécuter l'application
if __name__ == '__main__':
    app.run_server(debug=True)
