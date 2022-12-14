

import warnings
import time
import cv2
import mediapipe as mp
import keras
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)


# Grabbing the Holistic Model from Mediapipe and
# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5
)

# Initializing the drawng utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils

# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)

# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0

model = keras.models.load_model(r"/Users/atounsi/Documents/GitHub/Sing-Language-Translator/Training/model/best_model_dataflair3.h5")

label_dict = { 
    0:'A',
    1:'B',
    2:'C',
    3:'D',
    4:'E',
    5:'F',
    6:'G',
    7:'H',
    8:'I',
    9:'J',
    10:'K',
    11:'L',
    12:'M',
    13:'N',
    14:'O',
    15:'P',
    16:'Q',
    17:'R',
    18:'S',
    19:'T',
    20:'U',
    21:'V',
    22:'W',
    23:'X',
    24:'Y',
    25:'Z',
    26:'Blank'
}
while capture.isOpened():
    # capture frame by frame
    ret, frame = capture.read()
    # resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))
    # Converting the from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Making predictions using holistic model
    # To improve performance, optionally mark the image as not writable to
    # pass by reference.
    #image.flags.writable = False
    results = holistic_model.process(image)
    #image.flags.writable = True    
    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Drawing the Facial Landmarks
    '''mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACE_CONNECTIONS,
        mp_drawing.DrawingSpec(
            color=(255,0,255),
            thickness=1,
            circle_radius=1
        ),
        mp_drawing.DrawingSpec(
            color=(0,255,255),
            thickness=1,
            circle_radius=1
        )
	)'''

	# Drawing Right hand Land Marks
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
	)
    # Drawing Box around the right hand
    if results.right_hand_landmarks:
        # the right hand landmarks array
        right_hand_landmarks_x = [landmark.x for landmark in results.right_hand_landmarks.landmark]
        right_hand_landmarks_y = [landmark.y for landmark in results.right_hand_landmarks.landmark]
        # Calculating the width and height of the image
        width,height = image.shape[1],image.shape[0]
        # Calculating the min and max values of the right hand landmarks
        min_x = max(0,int(min(right_hand_landmarks_x)*width-0.1*width))
        max_x = min(int(max(right_hand_landmarks_x)*width+0.1*width),width-1)
        min_y = max(0,int(min(right_hand_landmarks_y)*height-0.1*height))
        max_y = min(int(max(right_hand_landmarks_y)*height+0.1*height),height-1)
        # Drawing the box around the right hand
        #print(width, height)
        #print(min_x, min_y)
        #print(max_x, max_y)
        image = cv2.rectangle(image, (min_x,min_y), (max_x,max_y), (0,255,0), 2)

        # cut array of right hand
        right_hand = image[min_y:max_y, min_x:max_x,:]
        # resize array of right hand
        right_hand = cv2.resize(right_hand, (64,64))
        # convert to gray

        predictions_right_hand = model.predict(np.expand_dims(right_hand, axis=0), verbose=0)

        idx_sign_right_hand = np.argmax(predictions_right_hand[0])
        print(idx_sign_right_hand, predictions_right_hand[0][idx_sign_right_hand])

        cv2.putText(img=image, 
                    text=f"{label_dict[idx_sign_right_hand]} : {predictions_right_hand[0][idx_sign_right_hand]*100:.2f}",
                    org=(min_x,min_y),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=3,
                    color=(255, 255, 255),
                    thickness=3)


    # Drawing Left hand Land Marks
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
	)

    # Drawing Box around the left hand
    if results.left_hand_landmarks:
        # the left hand landmarks array
        left_hand_landmarks_x = [landmark.x for landmark in results.left_hand_landmarks.landmark]
        left_hand_landmarks_y = [landmark.y for landmark in results.left_hand_landmarks.landmark]
        # Calculating the width and height of the image
        width,height = image.shape[1],image.shape[0]
        # Calculating the min and max values of the left hand landmarks
        min_x = max(0,int(min(left_hand_landmarks_x)*width-0.1*width))
        max_x = min(int(max(left_hand_landmarks_x)*width+0.1*width),width-1)
        min_y = max(0,int(min(left_hand_landmarks_y)*height-0.1*height))
        max_y = min(int(max(left_hand_landmarks_y)*height+0.1*height),height-1)
        # Drawing the box around the left hand
        image = cv2.rectangle(image, (min_x,min_y), (max_x,max_y), (255,0,0), 2)

        # cut array of left hand    
        left_hand = image[min_y:max_y, min_x:max_x,:]
        # resize array of left hand
        left_hand = cv2.resize(left_hand, (64,64))
        # convert to gray

        predictions_left_hand = model.predict(np.expand_dims(left_hand, axis=0), verbose=0)

        idx_sign_left_hand = np.argmax(predictions_left_hand[0])
        print(idx_sign_left_hand,predictions_left_hand[0][idx_sign_left_hand])

        cv2.putText(img=image,
                    text=f"{label_dict[idx_sign_left_hand]} : {predictions_left_hand[0][idx_sign_left_hand]*100:.2f}",
                    org=(min_x,min_y),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=2,
                    color=(255, 255, 255),
                    thickness=3)

	
    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime
	
    # Displaying FPS on the image
    cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    # Display the resulting image
    cv2.imshow("Facial and Hand Landmarks", image)

	# Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
