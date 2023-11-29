# space bar method 
import cv2
from yolov5.detect import run
import time
import os
import shutil

# Create a VideoCapture object to capture video from the default camera (0)
cap = cv2.VideoCapture(0)

# Load the Haarcascades classifier for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

last_time = time.time()

# Delete the children of the static folder
shutil.rmtree('./static', ignore_errors=True)
imgCount = 0
# Create the input folder
os.mkdir('./static')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for eye detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Check if all eyes are open
    all_eyes_open = len(eyes) == 2

    # Display the captured frame
    cv2.imshow('Webcam', frame)

    # Check for key press events
    key = cv2.waitKey(1) & 0xFF

    # Check if spacebar is pressed
    if key == ord(' '):
        if not all_eyes_open:
            print("Info: Not all eyes are open. Waiting for all eyes to be open...")
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()

                # Convert the frame to grayscale for eye detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect eyes in the frame
                eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

                # Check if all eyes are open
                if len(eyes) == 2:
                    print("Success: All eyes are open.")
                    break

                # Display the captured frame
                cv2.imshow('Webcam', frame)

                # Check for key press events
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        # Save the resulting frame
        resized_frame = cv2.resize(frame, (640, 640))
        cv2.imwrite('frame.jpg', resized_frame)

        # Run YOLOv5
        run(
            weights='BestWeight.pt',  # model.pt path(s)
            source="frame.jpg",  # file/dir/URL/glob, 0 for webcam
            save_txt=True,  # save results to *.txt
            project='output',
            nosave=True,  # do not save images/videos
        )

        # Read the output file
        # Check if output/exp/labels have a txt file called frame.txt
        if os.path.isfile('output/exp/labels/frame.txt'):
            # Open the file
            file = open('output/exp/labels/frame.txt', 'r')
            # Read all the lines
            lines = file.readlines()
            # Print the lines
            for line in lines:
                print(line)

            # TODO: Do custom stuff with the lines to save the image

            # Save the image
            cv2.imwrite(f'./static/frame{imgCount}.jpg', frame)
            imgCount += 1

            # Close the file
            file.close()

            shutil.rmtree('./output', ignore_errors=True)
        else:
            print("Error: YOLOv5 output file not found.")

    # Check for the 'q' key to exit the loop
    elif key == ord('q'):
        break

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()


