from function import *
from time import sleep

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                # ret, frame = cap.read()
                frame=cv2.imread('Image/{}/{}{}.jpg'.format(action,action,sequence+1))
                # print('Image/{}/{}{}.jpg'.format(action,action,sequence))
                
                # frame=cv2.imread('{}{}.png'.format(action,sequence))
                # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                # Make detections
                image, results = mediapipe_detection(frame, hands)
#                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(200)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    # cap.release()
    cv2.destroyAllWindows()

# This code is for collecting data and extracting keypoints for each frame in a video. 
# The data is being collected for different actions which are defined in the actions list.

# The first for loop creates directories for each action and sequence using os.makedirs function.

# The mediapipe model is being used for detecting and extracting hand landmarks in each frame of the video.

# Then there is a nested for loop that loops through each action, sequence, and frame in the video. 
# For each frame, the extract_keypoints function is called to extract the landmarks from the image. 
# Then, the landmarks are saved in a .npy file with the corresponding action, sequence, and frame number.

# The keypoints are saved in the following directory format:
# /Data/action/sequence/frame_num.npy

# While saving the keypoints, the image is also displayed on the screen using cv2.imshow and waits for 10ms before showing the next frame.

# The loop runs for a total of no_sequences * sequence_length frames for each action. 
# If the user presses the "q" key, the loop breaks and the window is closed.
