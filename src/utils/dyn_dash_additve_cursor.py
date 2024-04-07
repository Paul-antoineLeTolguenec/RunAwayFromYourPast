import plotly.graph_objs as go
import numpy as np

# Initialisation de la figure
fig = go.Figure()

# Fonction pour générer des points aléatoires
def generate_points(num_points: int):
    return np.random.rand(num_points), np.random.rand(num_points), np.random.rand(num_points)

# Nombre initial de points et nombre d'étapes d'animation
num_initial_points = 10
num_frames = 10
new_points_per_frame = 5

# Points de départ
x_base, y_base, z_base = generate_points(num_initial_points)
indices_base = list(range(num_initial_points))

# Ajouter les points de départ à la figure avec une échelle de couleur
fig.add_trace(go.Scatter3d(
    x=x_base, y=y_base, z=z_base,
    mode='markers',
    marker=dict(size=5, color=indices_base, colorscale='Viridis', colorbar=dict(title='Point Order'))
))

# Préparation des listes pour stocker l'ensemble cumulatif de points et leurs indices
x_cumulative, y_cumulative, z_cumulative = x_base.tolist(), y_base.tolist(), z_base.tolist()
indices_cumulative = indices_base[:]

# Création des frames supplémentaires avec ajout progressif de points
frames = []
for i in range(1, num_frames):
    # Générer de nouveaux points
    x_new, y_new, z_new = generate_points(new_points_per_frame)
    new_indices = list(range(len(indices_cumulative), len(indices_cumulative) + len(x_new)))
    
    # Ajouter les nouveaux points et indices aux listes cumulatives
    x_cumulative.extend(x_new)
    y_cumulative.extend(y_new)
    z_cumulative.extend(z_new)
    indices_cumulative.extend(new_indices)

    # Créer un nouveau frame avec l'ensemble cumulatif de points et gradient de couleur
    frame = go.Frame(data=[go.Scatter3d(
        x=x_cumulative, y=y_cumulative, z=z_cumulative,
        mode='markers',
        marker=dict(size=5, color=indices_cumulative, colorscale='Viridis')
    )], name=str(i))
    frames.append(frame)

fig.frames = frames

# Ajout du slider
sliders = [{
    'steps': [
        {
            'method': 'animate',
            'label': str(i),
            'args': [[frame.name], dict(frame=dict(duration=0, redraw=True), mode='immediate', transition=dict(duration=0))]
        } for i, frame in enumerate(fig.frames)
    ],
    'transition': {'duration': 0},
    'x': 0.1, 'len': 0.9, 'xanchor': 'left', 'y': 0, 'yanchor': 'top'
}]

fig.update_layout(
    sliders=sliders,
    scene=dict(
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        zaxis=dict(range=[0, 1])
    )
)

# Sauvegarder la figure dans un fichier HTML
fig.write_html('animated_3d_points_additive_with_slider.html')
