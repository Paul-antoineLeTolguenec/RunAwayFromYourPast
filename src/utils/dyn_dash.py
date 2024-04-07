import plotly.graph_objs as go
import numpy as np

# Initialisation de la figure
fig = go.Figure()

# Fonction pour générer des points aléatoires
def generate_points(num_points: int):
    return np.random.rand(num_points), np.random.rand(num_points), np.random.rand(num_points)

# Nombre de points initiaux et nombre d'états (frames)
num_initial_points = 10
num_frames = 10

# Création du premier état (frame)
x, y, z = generate_points(num_initial_points)
fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5)))

# Ajout des frames supplémentaires
frames = []
for i in range(1, num_frames):
    x, y, z = generate_points(num_initial_points + i * 5)  # Augmentation du nombre de points
    frame = go.Frame(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5))], name=str(i))
    frames.append(frame)

fig.frames = frames

# Ajout des boutons de contrôle de l'animation
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]
                ),
            ],
        )
    ]
)

# Sauvegarder la figure dans un fichier HTML
fig.write_html('animated_3d_points.html')
