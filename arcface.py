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
        
        self.threshold = 0.6
        self.face_register = {}
    
    def register_face(self, name, image, crop = False):
        
        if crop:
            image = utils.crop_face(image, utils.get_face_bb_opencv(image))

        embedding = self.predict(image)
        self.face_register[name] = embedding

    def recognize_face(self, image):
        embedding = self.predict(image)

        # key will be name, value will be cosine similarity
        match_faces = {}

        for name in self.face_register.keys():
            match, cosine_similarity = self.match(embedding, self.face_register[name])
            if match:
                match_faces[name] = cosine_similarity
        
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

        threshold = 0.5

        # # Load LFW dataset
        # import tensorflow_datasets as tfds
        # lfw = tfds.load("lfw")

        # # Calculate embeddings for LFW dataset
        # lfw_embeddings = {}
        # for name in lfw.keys():
        #     lfw_embeddings[name] = []
        #     for image in lfw[name]:
        #         embedding = model.predict(cv2.resize(image, (112, 112)).astype(np.float32) / 255.0)
        #         lfw_embeddings[name].append(embedding)

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
                similarity = self.cosine(input_embeddings[i], input_embeddings[j])
                similarities.append(similarity)
        
        # Plot histogram of similarities
        import matplotlib.pyplot as plt
        plt.hist(similarities, bins=100)
        plt.show()

        # Choose threshold that separates positive and negative pairs well
        threshold = np.mean(similarities) - np.std(similarities)
        
        return threshold