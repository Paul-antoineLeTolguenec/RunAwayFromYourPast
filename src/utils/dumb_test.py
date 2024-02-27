import torch

# # Définition du seuil
# seuil = 0.1

# # Fonction de rappel pour le hook
# def hook_fn_gradients(grad):
#     print('shape gradient:', grad.shape)
#     print('gradient:', grad)
#     # Créer un masque où les gradients sont désactivés si la sortie est inférieure au seuil
#     masque = grad >= seuil
#     # Appliquer le masque sur le gradient, en mettant à 0 les gradients des sorties inférieures au seuil
#     return grad * masque.float()

# # Modèle simple pour l'exemple
# class SimpleModel(torch.nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.linear = torch.nn.Linear(10, 1)  # Exemple de couche

#     def forward(self, x):
#         x = self.linear(x)
#         return x

# # Créer une instance du modèle
# model = SimpleModel()

# # Effectuer le passage forward
# x = torch.randn(3, 10, requires_grad=True)  # Batch de taille 5
# output = model(x)
# print('shape output:', output.shape)
# print('output:', output)

# # Attacher le hook au tenseur de sortie pour modifier son gradient
# hook = output.register_hook(hook_fn_gradients)

# # Exemple de calcul de perte et de rétropropagation
# loss = output.sum()  # Utilisation d'une fonction de perte simple pour l'exemple
# loss.backward()

# # Vérification des gradients après application du hook
# print('shape gradient:', x.grad.shape)
# print('gradient:', x.grad)

# # Suppression du hook pour éviter des effets de bord sur d'autres calculs
# hook.remove()

a = torch.randn(3, 10)
seuil = 0.1
mask = a >= seuil
print('a:', a)
print('mask:', mask)