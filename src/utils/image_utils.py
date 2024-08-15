from PIL import Image
import numpy as np

def resize_image(image, new_width, new_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((new_width, new_height), Image.ANTIALIAS)
    return np.array(resized_image)