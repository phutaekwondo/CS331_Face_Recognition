import tensorflow as tf
import numpy as np
from deepface import DeepFace
import utils
import cv2

class ArcFaceModel:
    def __init__(self, model_path = None):
        if model_path == None:
            self.model = DeepFace.build_model("ArcFace")
        else:
            self.model = tf.keras.models.load_model(model_path)
        
        self.threshold = 0.4
        self.face_register = {}
    
    def register_face(self, name, images):
        
        # if crop:
        #     image = utils.crop_face(image, utils.get_face_bb_opencv(image))
        embeddings = []
        for image in images:
            embedding = self.predict(image)
            embeddings.append(embedding)

        self.face_register[name] = embeddings

    def recognize_face(self, image):
        embedding = self.predict(image)

        # key will be name, value will be cosine similarity
        match_faces = {}

        for name in self.face_register.keys():

            highest_cosine_similarity = 0
            for register_embedding in self.face_register[name]:
                match, cosine_similarity = self.match(embedding, register_embedding)
                
                if match and cosine_similarity > highest_cosine_similarity:
                    highest_cosine_similarity = cosine_similarity
                
            match_faces[name] = highest_cosine_similarity
        
        if len(match_faces) > 0:
            # return the name with the highest cosine similarity
            return max(match_faces, key=match_faces.get)

        return None

    def summary(self):
        self.model.summary()

    def predict(self, image):
        image = cv2.resize(image, (112, 112))
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = image / 255.0
        embedding = self.model.predict(image)
        return embedding

    def match(self, embedding1, embedding2):
        cosine_similarity = -1 * tf.keras.losses.cosine_similarity(embedding1, embedding2)
        
        try:
            cosine_similarity = cosine_similarity.numpy()[0]
        except:
            cosine_similarity = 0
        
        print('cosine similarity:', cosine_similarity)

        return cosine_similarity > self.threshold, cosine_similarity
    
    # def cosine(self, embedding1, embedding2):
       
    #     cosine = -1 * tf.keras.losses.cosine_similarity(embedding1, embedding2)
        
    #     try:
    #         cosine = cosine.numpy()[0]
    #     except:
    #         cosine = 0
        
    #     return cosine
    
    def set_threshold(self, threshold):
        self.threshold = threshold
        return self.threshold
    
    def find_good_threshold(self, list_image):

        threshold = self.threshold

        # Calculate embeddings for input images
        input_embeddings = []
        for image in list_image:
            print(image.shape)
            embedding = self.predict(image)
            input_embeddings.append(embedding)

        # Calculate cosine similarity between each pair of embeddings
        similarities = []
        for i in range(len(input_embeddings)):
            for j in range(i+1, len(input_embeddings)):
                _, similarity = self.match(input_embeddings[i], input_embeddings[j])
                similarities.append(similarity)
        
        # Plot histogram of similarities
        import matplotlib.pyplot as plt
        plt.hist(similarities, bins=100)
        plt.show()

        # Choose threshold that separates positive and negative pairs well
        threshold = np.mean(similarities) - np.std(similarities)
        
        return threshold