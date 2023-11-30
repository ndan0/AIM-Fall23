import cv2
from yolov5.detect import run, parse_opt
import time
import os
import shutil

# Create a VideoCapture object to capture video from the default camera (0)
cap = cv2.VideoCapture(0)

last_time = time.time()

# Delete the children of the static folder
shutil.rmtree('./static', ignore_errors=True)
imgCount = 0
# Create the input folder
os.mkdir('./static')

spacePressed = False


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Save the resulting frame
    resized_frame = cv2.resize(frame, (640, 640))
    cv2.imwrite('frame.jpg', resized_frame)
    
    # Display the captured frame
    cv2.imshow('Webcam', frame)
    
    # Run YOLOv5 every 1 second
    if spacePressed and time.time() - last_time > 1:
        last_time = time.time()
        run(
            weights='best.pt',  # model.pt path(s)
            source="frame.jpg",  # file/dir/URL/glob, 0 for webcam
            save_txt=True,  # save results to *.txt
            project = 'output',
            #nosave=True,  # do not save images/videos
        )
        # Read the output file
        # CHeck if output/exp/labels have a txt file called frame.txt
        if os.path.isfile('output/exp/labels/frame.txt'):
            # Open the file
            file = open('output/exp/labels/frame.txt', 'r')
            # Read all the lines
            lines = file.readlines()
            # Print the lines
            eyes_open = False
            for line in lines:
                print(line)
                
            #TODO: Do custom stuff with the lines to save the image
                firstChar = line[0]
                if firstChar == "2":
                    eyes_closed = True
                
            # Save the image
            if eyes_closed:
                cv2.imwrite(f'./static/frame{imgCount}.jpg', frame)
                imgCount += 1
                spacePressed = False
            

            # Close the file
            file.close()
        #spacePressed = False
        shutil.rmtree('./output', ignore_errors=True)

    # Check for the 'q' key to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        spacePressed = True
        print("space pressed")
        

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()
