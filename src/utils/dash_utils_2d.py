import plotly.graph_objs as go
import os 
import numpy as np

def initialize_figure_2d(render_settings = {'x_lim': [-1, 1], 'y_lim': [-1, 1]}) -> go.Figure:
    """Initialise la figure Plotly en 2D avec des paramètres de base.

    Returns:
        go.Figure: La figure Plotly initialisée en 2D.
    """
    fig = go.Figure()
    # fig.update_layout(xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    fig.update_layout(xaxis=dict(range=render_settings['x_lim']), yaxis=dict(range=render_settings['y_lim']))
    return fig


def add_frame_to_figure_2d(fig: go.Figure, x: np.ndarray, y: np.ndarray, indices: np.ndarray, frame_id: int, keep_previous: bool = False) -> None:
    """Ajoute ou met à jour une frame dans la figure en 2D avec les points et leur gradient de couleur.

    Args:
        fig (go.Figure): La figure à mettre à jour.
        x (np.ndarray): Les coordonnées x des points.
        y (np.ndarray): Les coordonnées y des points.
        indices (np.ndarray): Les indices des points pour le gradient de couleur.
        frame_id (int): Identifiant unique de la frame.
        keep_previous (bool, optional): Si `False`, chaque frame représente seulement son propre ensemble de points. Si `True`, la frame ajoute ses points à ceux de toutes les frames précédentes. Defaults to True.
    """
    marker = dict(size=5, color=indices, colorscale='Viridis', cmin=np.min(indices), cmax=np.max(indices), colorbar=dict(title='Point Order') if frame_id == 0 else None)

    if frame_id == 0 :
        trace = go.Scatter(x=x, y=y, mode='markers', marker=marker, showlegend=False)
        fig.add_trace(trace)

    if frame_id > 0 and keep_previous:
        for trace in fig.data:
            trace['x'] = np.concatenate((trace['x'], x)).copy()
            trace['y'] = np.concatenate((trace['y'], y)).copy()
            trace['marker']['color'] = np.concatenate((trace['marker']['color'], indices))
    if keep_previous :
        frame_traces = [dict(type='scatter', x=trace['x'], y=trace['y'], mode='markers', marker=trace['marker'])]
    else :
        frame_traces = [dict(type='scatter', x=x, y=y, mode='markers', marker=marker)]
    frame = go.Frame(data=frame_traces, name=str(frame_id))

    if not hasattr(fig, 'frames') or fig.frames is None:
        fig.frames = []

    fig.frames += (frame,)


def create_html_2d(fig: go.Figure, filename: str) -> None:
    """Sauvegarde la figure 2D avec animation dans un fichier HTML.

    Args:
        fig (go.Figure): La figure à sauvegarder avec les frames d'animation.
        filename (str): Le nom du fichier HTML de sortie.
    """
    sliders = [{
        'pad': {"t": 30},
        'steps': [
            {
                'method': 'animate',
                'label': str(frame.name),
                'args': [[frame.name], dict(frame=dict(duration=0, redraw=True), mode='immediate', transition=dict(duration=0))]
            } for frame in fig.frames
        ]
    }]
    # html folder 
    if not os.path.exists('html'):
                os.makedirs('html')
    fig.update_layout(sliders=sliders)
    fig.write_html(f'html/{filename}.html')

def add_walls_to_figure_2d(fig: go.Figure) -> None:
    """Ajoute des murs (lignes) à la figure 2D.

    Args:
        fig (go.Figure): La figure à mettre à jour.
    """
    # Mur horizontal
    fig.add_trace(go.Scatter(
        x=[0, 1],  # Coordonnées x de début et de fin
        y=[0.5, 0.5],  # Coordonnées y de début et de fin (même valeur pour une ligne horizontale)
        mode='lines',  # Mode 'lines' pour dessiner une ligne
        line=dict(color='RoyalBlue', width=5),  # Personnalisation de la ligne
        showlegend=False  # Ne pas afficher ce tracé dans la légende
    ))
    # Mur vertical
    fig.add_trace(go.Scatter(
        x=[0.5, 0.5],  # Coordonnées x de début et de fin (même valeur pour une ligne verticale)
        y=[0, 1],  # Coordonnées y de début et de fin
        mode='lines',
        line=dict(color='Crimson', width=2),
        showlegend=False
    ))


if __name__ == '__main__':
    # Initialiser la figure en 2D
    fig_2d = initialize_figure_2d()
    # add_walls_to_figure_2d(fig_2d)
    # add_frame_to_figure_2d(fig_2d, np.random.rand(10), np.random.rand(10), np.linspace(0, 1, 10), 0)
    # Générer des données de points et des indices pour la couleur, puis ajouter plusieurs frames à la figure
    for frame_id in range(10):  # Nombre de frames à générer
        num_points = 10   # Augmente le nombre de points pour chaque frame
        x, y = np.random.rand(2, num_points)  # Générer des coordonnées aléatoires pour x et y
        indices = np.linspace(start=0, stop=1, num=num_points)  # Générer des indices pour le gradient de couleur

        # Ajouter une nouvelle frame à la figure en 2D
        add_frame_to_figure_2d(fig_2d, x, y, indices, frame_id, keep_previous=True)

    # Sauvegarder la figure en 2D dans un fichier HTML avec un slider pour contrôler les frames
    create_html_2d(fig_2d, '2d_points_with_slider.html')
