import cv2
"Class to detect face from captured frame"


class FaceDetector:

    def __init__(self, face_casc='classifiers/haarcascade_frontalface_default.xml',
                 scale_factor=4):
        # assign given scale factor to global variable
        self.scale_factor = scale_factor

        # assign cv2 Classifier to global variable
        self.face_casc = cv2.CascadeClassifier(face_casc)

    # function to detect, extract and return face from captured frame
    def detect_face(self, frame):
        capFrame = cv2.cvtColor(cv2.resize(
                                frame,
                                (0, 0),
                                fx=1.0 / self.scale_factor,
                                fy=1.0 / self.scale_factor),
                                cv2.COLOR_RGB2GRAY)
        face = self.face_casc.detectMultiScale(
                                    capFrame,
                                    scaleFactor=1.1,
                                    minNeighbors=3,
                                    flags=cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT) * self.scale_factor

        # if face is detected draw a rectangle on a face and extract it
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            head = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY)
            head = cv2.resize(head, dsize=(200, 200))
            return True, frame, head, (x, y)
        return False, frame, None, (0, 0)
