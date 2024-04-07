import plotly.graph_objs as go
import numpy as np
import os

def initialize_figure_3d(render_settings = {'x_lim': [-1, 1], 'y_lim': [-1, 1], 'z_lim': [-1, 1]}) -> go.Figure:
    """Initialise la figure Plotly avec des paramètres de base.

    Returns:
        go.Figure: La figure Plotly initialisée.
    """
    fig = go.Figure()
    fig.update_layout(scene=dict(xaxis=dict(range=render_settings['x_lim']), yaxis=dict(range=render_settings['y_lim']), zaxis=dict(range=render_settings['z_lim'])))
    return fig






def add_frame_to_figure_3d(fig: go.Figure, x: np.ndarray, y: np.ndarray, z: np.ndarray, indices: np.ndarray, frame_id: int, keep_previous: bool = True) -> None:
    """Ajoute ou met à jour une frame dans la figure avec les points et leur gradient de couleur, en mettant à jour la colorbar à chaque frame.

    Args:
        fig (go.Figure): La figure à mettre à jour.
        x (np.ndarray): Les coordonnées x des points.
        y (np.ndarray): Les coordonnées y des points.
        z (np.ndarray): Les coordonnées z des points.
        indices (np.ndarray): Les indices des points pour le gradient de couleur.
        frame_id (int): Identifiant unique de la frame.
        keep_previous (bool, optional): Si `False`, chaque frame représente seulement son propre ensemble de points. Si `True`, la frame ajoute ses points à ceux de toutes les frames précédentes. Defaults to True.
    """

    # Définir les propriétés du marqueur, y compris la mise à jour de la colorbar pour chaque frame
    marker = dict(
        size=5,
        color=indices,
        colorscale='Viridis',
        cmin=np.min(indices),  # Définir le minimum de la colorbar en fonction des données actuelles
        cmax=np.max(indices),  # Définir le maximum de la colorbar en fonction des données actuelles
        colorbar=dict(title='Point Order') if frame_id == 0 else None  # Afficher la colorbar seulement pour la première frame
    )

    if frame_id == 0 :
        # Pour la première frame ou quand keep_previous est False, ajouter une nouvelle trace avec la colorbar
        trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=marker, showlegend=False)
        fig.add_trace(trace)

    # Pour les frames suivantes ou si keep_previous est True, mettre à jour les données des traces existantes sans ajouter une nouvelle colorbar
    if frame_id > 0 and keep_previous:
        for trace in fig.data:
            trace['x'] = np.concatenate((trace['x'], x)).copy()
            trace['y'] = np.concatenate((trace['y'], y)).copy()
            trace['z'] = np.concatenate((trace['z'], z)).copy()
            trace['marker']['color'] = np.concatenate((trace['marker']['color'], indices))

    # Créer la frame actuelle
    if keep_previous : 
        frame_traces = [dict(type='scatter3d', x=trace['x'], y=trace['y'], z=trace['z'], mode='markers', marker=trace['marker'])]  
    else : 
        frame_traces = [dict(type='scatter3d', x=x, y=y, z=z, mode='markers', marker=marker)]
    frame = go.Frame(data=frame_traces, name=str(frame_id))

    if not hasattr(fig, 'frames') or fig.frames is None:
        fig.frames = []

    fig.frames += (frame,)



def create_html_3d(fig: go.Figure, filename: str) -> None:
    """Sauvegarde la figure avec animation dans un fichier HTML.

    Args:
        fig (go.Figure): La figure à sauvegarder avec les frames d'animation.
        filename (str): Le nom du fichier HTML de sortie.
    """
    # Configuration des sliders pour le contrôle de l'animation
    sliders = [{
        'pad': {"t": 30},
        'steps': [
            {
                'method': 'animate',
                'label': str(frame.name),
                'args': [[frame.name], dict(frame=dict(duration=0, redraw=True), mode='immediate', transition=dict(duration=0))]
            } for (frame, idx) in zip(fig.frames, range(len(fig.frames)))
        ]
    }]
    # html folder 
    if not os.path.exists('html'):
                os.makedirs('html')
    fig.update_layout(sliders=sliders)
    fig.write_html(f'html/{filename}.html')
    


if __name__ == '__main__':
    # Exemple d'utilisation
    fig = initialize_figure_3d()

    # Génération des données de points et des indices pour la couleur, puis ajout de plusieurs frames
    for frame_id in range(10):
        num_points = 10 * (1)
        x, y, z = np.random.rand(3, num_points)
        indices = np.arange(num_points)
        add_frame_to_figure_3d(fig, x, y, z, indices, frame_id, keep_previous=False)

    # Sauvegarde de la figure dans un fichier HTML avec un slider pour contrôler les frames
    create_html_3d(fig, '3d_points_with_slider.html')


