import os
import torch
from torch.utils.data import Dataset as BaseDataset # Lo renombramos para que no choque
from PIL import Image
import numpy as np

class CloudDataset(BaseDataset):
    
    def __init__(self, folder_path, txt_path, transform=None):
        self.folder_path = folder_path
        self.txt_path = txt_path
        self.transform = transform

        # 1. Guardamos los nombres de las imágenes
        self.image_names = []

        # Leemos el .txt
        self.labels = {}
        with open(self.txt_path, 'r') as f:
            for line in f:
                partes = line.strip().split(';')
                if len(partes) >= 2:
                    try:
                        nombre = partes[0].strip()
                        # Intentamos convertir la segunda columna a número
                        altura = float(partes[1].strip()) 
                        
                        # Si es un número, verificamos que la imagen exista y la añadimos
                        if os.path.exists(os.path.join(self.folder_path, nombre)):
                            self.image_names.append(nombre)
                            self.labels[nombre] = altura
                    except ValueError:
                        # Si no es un número ignora el dato y pasa al siguiente
                        continue 

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        img_name = self.image_names[index]
        image_path = os.path.join(self.folder_path, img_name) 
    
        image = Image.open(image_path).convert('RGB')

        # 3. Aplicamos transformaciones
        if self.transform:
            image = self.transform(image)
        else:
            # Si no hay transformaciones, lo pasamos a tensor manualmente
            image = np.array(image) / 255.0
            image = image.astype(np.float32)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)

        # Pasamos el valor de la altura a la clase
        altura = self.labels[img_name]
        if altura < 2000.0:
            clase = 0 # rango bajo
        elif altura < 6000.0:
            clase = 1 # rango medio
        else:
            clase = 2 # rango alto
        
        label = torch.tensor(clase, dtype=torch.long)

        return image, label






#CÓDIGO VIEJO
# from torch.utils.data import Dataset #Es la clase base de PyTorch que necesitas "heredar" para crear tu propio cargador de datos.
# import os #interactuar con el sistema operativo, como leer los nombres de los archivos en una carpeta o unir rutas de carpetas
# import torch
# from PIL import Image #librería estándar de Python para abrir y manipular imágenes
# import numpy as np #trabajar con matrices numéricas

# class dataset_train(Dataset):
#     def __init__(self, image_path, txt_path):
#         self.images_path=image_path #no es necesario definirlo dos veces
#         self.txt_path=txt_path

#         #lista de textos (strings), ordenada alfabéticamente, que contiene únicamente los nombres de los archivos de imagen que están dentro de esa carpeta.
#         self.images_path=sorted([f for f in os.listdir(image_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

#     def __len__(self): #Le dice a PyTorch cuántas imágenes hay en total.
#         return len(self.images_path)
    
#     def __getitem__(self, index):
#         image_path= os.path.join(self.images_path, self.images_path[index]) 
    
#         # Abrimos imagen, la pasamos a array y normalizamos
#         image= np.array(Image.open(image_path).convert('RGB')) #abrimos imagen
#         image=image/255 #normalizamos
#         image=image.astype(np.float32)

#         # Pasamos de (Alto, Ancho, Canales) a (Canales, Alto, Ancho). Así PyTorch la lee bien
#         image = np.transpose(image, (2, 0, 1))
        
#         # Convertimos la imagen de NumPy a un Tensor de PyTorch
#         image_tensor = torch.from_numpy(image)

#         txt= os.open(self.txt_path)

#         # 1. Extraemos solo el nombre del archivo de la ruta completa
#         nombre_imagen_buscada = os.path.basename(image_path)
#         altura = None

#         for line in txt:
#             # Asumimos que el separador es una coma o un espacio
#             partes = line.strip().split(';') # Si están separados por comas, usa split(',')
            
#             # Comprobamos que la línea tenga al menos 2 columnas. No lo necesito pero por si acaso
#             if len(partes) >= 2:
#                 nombre_txt = partes[0].strip()
                
#                 # 3. Si el nombre en la línea del txt coincide con nuestra imagen
#                 if nombre_txt == nombre_imagen_buscada:
#                     # Cogemos la columna 1 (el segundo elemento), lo pasamos a float
#                     altura = float(partes[1].strip())
#                     break # Encontramos la altura, paramos de buscar en el txt

#         # 4. Convertimos la altura al tensor que requiere PyTorch para regresión
#         height = torch.tensor(altura, dtype=torch.float32)

        

#         return image_tensor, height

# #como otra opción podemos utilizar las funciones de augment.py para modificar ligeramente las imagenes (rotaciones, saturación) y entrenar más a la IA
