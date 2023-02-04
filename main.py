from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

class Model:
    
    def __init__(self,):
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)
        
        # Load the model
        self.model = load_model("keras_model.h5", compile=False)

        # Load the labels
        self.class_names = open("labels.txt", "r").readlines()
    
    
    def predict(self, file_path):
        
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        image = Image.open(file_path).convert("RGB")

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.BICUBIC)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = self.model.predict(data)
        index = np.argmax(prediction)
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        # print("Class:", class_name[2:], end="")
        # print("Confidence Score:", confidence_score)
        return class_name[2:], confidence_score
        
        

if __name__ == "__main__":
    obj = Model()
    class_name, confidence_score = obj.predict("240.jpg")
    print(class_name, confidence_score)
        







