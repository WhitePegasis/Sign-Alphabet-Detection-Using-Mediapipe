import cv2
import numpy as np
import os
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


def extract_keypoints(results):
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3)
        return(np.concatenate([rh]))
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

no_sequences = 30

sequence_length = 30

# The code is using the MediaPipe library to detect and track hand landmarks in images.

# First, the necessary libraries are imported, including OpenCV, NumPy, os, and MediaPipe.

# The function mediapipe_detection() takes an image and a MediaPipe model as inputs, converts the image from BGR to RGB format, processes the image using the model, and then converts the image back to BGR format before returning the image and the results.

# The function draw_styled_landmarks() takes an image and results as inputs and uses MediaPipe's draw_landmarks() function to draw the hand landmarks and connections on the image.

# The function extract_keypoints() takes the results as input and extracts the 3D coordinates of the hand landmarks, if detected.

# The code defines a DATA_PATH variable for storing the exported data, which will be in the form of NumPy arrays.

# The code also defines an actions array, which contains the possible actions that the model will be trained to recognize.

# no_sequences and sequence_length variables are used to define the length and number of sequences that will be generated during training.

# Overall, the code sets up the necessary functions and variables for training a hand gesture recognition model using MediaPipe and NumPy.

