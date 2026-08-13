import cv2

def test_webcam():
    # Initializing the webcam (0 usually represents the default built-in camera)
    capture = cv2.VideoCapture(0)

    #Checking if the webcam opened correctly
    if not capture.isOpened():
        print("Error: Webcam could not be opened.")
        return

    #Printing message for the webcam opening correctly
    print("Webcam feed opened successfully. Press 'q' to quit.")

    while True:
        #Capturing data into the webcam frame by frame
        return_value, frame = capture.read()

        #return_value is true if the frames were successfully read
        if not return_value:
            print("Error: Could not receive frame. Exiting...")
            break

        #Displaying the resulting frame(s) if it/they were successfully read
        cv2.imshow('ASL Interpreter - Webcam Test', frame)

        #Waiting 1 millisecond and checking if the user pressed 'q' to quit using the webcam
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

    #After everything is done, disconnecting the interpreter system from the webcam and closing every graphic window opened by OpenCV
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_webcam()