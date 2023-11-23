import cv2
from yolov5.detect import run, parse_opt
import time

# Create a VideoCapture object to capture video from the default camera (0)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Save the resulting frame
    resized_frame = cv2.resize(frame, (640, 640))
    cv2.imwrite('frame.jpg', resized_frame)
    

    run(
        weights='BestWeight.pt',  # model.pt path(s)
        source="frame.jpg",  # file/dir/URL/glob, 0 for webcam
        save_txt=True,  # save results to *.txt
        project = 'output',
        nosave=True,  # do not save images/videos
    )
    
    time.sleep(1)
    
    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()
