import cv2
import numpy as np

# Face classifiers / 얼굴 분류
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Grab Webcam feed / 웹캠
webcam = cv2.VideoCapture(0)

# Show the current frame / 웹캠실행 및 얼굴 인식
while True:
    
    # Read the current frame from the webcam video stream / 영상 인식
    successful_frame_read, frame = webcam.read()
    
    # If there's an error, abort / 오류 발생시 무시
    if not successful_frame_read:
        break

    # Change to grayscale / 회색조 변환
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces first / 얼굴 인식
    faces = face_detector.detectMultiScale(frame_grayscale)

    # Run face detection within each of those faces / 판별기 실행
    for (x, y, w, h) in faces:

        # Draw a rectangle around the face / 얼굴 주변 사각형
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 50), 4)

        # Get the sub frame (using numpy N-dimensional array slicing) / 얼굴 프레임 (넘파이 N-dimensional array slicing 사용)
        the_face = frame[y:y+h, x:x+w]

        # Change to grayscale / 회색조 변환
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor = 1.7, minNeighbors = 20)

        # Find all smiles in the face / 얼굴에서 미소 탐색
        #for (x_, y_, w_, h_) in smiles:

            # Draw a rectangle around the smile / 미소 주변 사각형
            #cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (50, 50, 200), 4)    

        # Label this face as smiling / 얼굴을 미소로 라벨링
        if len(smiles) > 0:
            cv2.putText(frame, 'smiling :)', (x, y + h + 40), fontScale = 3, fontFace=cv2.FONT_HERSHEY_PLAIN, color = (255, 255, 255))

     # Show the current frame / 현재 프레임 표시
    title = 'Let\'s Smile!'
    cv2.imshow(title, frame)

    # Display / 표시
    cv2.waitKey(1)

# Cleanup / 종료
webcam.release()
cv2.destroyAllWindows()

# Code ran without errors / 코드 실행 중 에러 발생시 메시지 
print("Code Completed")
