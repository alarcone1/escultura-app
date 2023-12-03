from clarifai.rest import ClarifaiApp
import requests
import openai

def procesar_imagen(api_key, image_url):
   """
      Procesa una imagen utilizando la API de Clarifai.

      Argumentos:
      api_key (str): La clave de la API de Clarifai.
      image_url (str): La URL de la imagen a procesar.

      Retorna:
      list: Una lista de conceptos detectados en la imagen.
   """
   app = ClarifaiApp(api_key=api_key)
   image_data = requests.get(image_url).content
   response = app.public_models.general_model.predict_by_url(url=image_url)
   concepts = response['outputs'][0]['data']['concepts']
   return concepts

def mostrar_listado_objetos(concepts):
   """
      Muestra el listado de objetos detectados en la imagen.

      Argumentos:
      concepts (list): Una lista de conceptos detectados en la imagen.

      Retorna:
      None
   """
   for i, concept in enumerate(concepts, 1):
      print(f"{i}. {concept['name']}")


def generar_imagen_escultura(prompt):
   """
      Genera una imagen de escultura basada en el prompt utilizando la API de OpenAI.

      Argumentos:
      prompt (str): El prompt para generar la imagen de la escultura.
      
      Retorna:
      str: La ruta de la imagen generada.
   """
   openai.api_key = 'TU_API_KEY'  # Reemplaza 'TU_API_KEY' con tu clave de API de OpenAI

   response = openai.Completion.create(
      engine="davinci-003",
      prompt=prompt,
      max_tokens=150
   )

   imagen_generada = response.choices[0].text.strip()
   return imagen_generada

# Interacción con el usuario
def main():
   api_key = input("Ingresa tu clave de API de Clarifai: ")
   image_url = input("Ingresa la URL de la imagen a procesar: ")

   # Procesar la imagen
   concepts = procesar_imagen(api_key, image_url)

   # Mostrar el listado de objetos
   print("Listado de objetos detectados en la imagen:")
   mostrar_listado_objetos(concepts)

   # Generar la imagen de la escultura
   prompt = "Se ha construido una imagen que contiene una escultura con los siguientes objetos: "
   for concept in concepts:
      prompt += concept['name'] + ", "
   imagen_generada = generar_imagen_escultura(prompt)

   print("La imagen de la escultura generada es:")
   print(imagen_generada)

if __name__ == "__main__":
   main()

