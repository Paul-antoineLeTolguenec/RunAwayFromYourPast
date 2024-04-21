import numpy as np

class CompatibleRandomGenerator:
    def __init__(self, generator):
        self.generator = generator

    def __getattr__(self, name):
        # Cette méthode est appelée si l'attribut n'existe pas normalement
        return getattr(self.generator, name)

    def randint(self, low, high=None, size=None, dtype=int):
        # Implémente randint en utilisant integers
        return self.generator.integers(low, high=high, size=size, dtype=dtype)
    