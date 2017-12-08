import cv2
import numpy as np
import wx
from os import path
from gui import BaseWindow
from FaceDetector import FaceDetector
import LoadData
import MultiLayerPerceptron


class MainLayout(BaseWindow):

    def init_layout(self):
        self.Bind(wx.EVT_CLOSE, self._on_exit)

    def init_algorithm(
            self,
            save_training_file = 'datasets/faces_training.pkl',
            load_mlp = 'datasets/mlp.xml',
            face_casc = 'classifiers/haarcascade_frontalface_default.xml'):
        self.training_file = save_training_file
        self.loadPreprocessedData = "datasets\\faces_preprocessed.pkl"
        self.loadMLP = load_mlp
        self.face = FaceDetector(face_casc)
        self.head=None

        if path.isfile(self.loadPreprocessedData):
            (_, yTrain), (_, yTest), V, m = LoadData.loadPreprocessedData(self.loadPreprocessedData)
            self.V = V
            self.m = m
            self.labels = np.unique(np.hstack((yTrain, yTest)))

        if path.isfile(load_mlp):
            layerSizes = np.array([self.V.shape[1],
                                   len(self.labels)])
            self.MultiLayerPerceptron = MultiLayerPerceptron.MLP(layerSizes, self.labels)
            self.MultiLayerPerceptron.loadFromFile(load_mlp)

    def processFrame(self, frame):

        success, frame, self.head, (x, y) = self.face.detect_face(frame)
        if success:
            X, _, _ = LoadData.extractFeatures([self.head.flatten()], self.V, self.m)
            label = self.MultiLayerPerceptron.predict(np.array(X))[0]
            cv2.putText(frame, str(label), (x, y - 20),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        return frame

    def on_exit(self):
        self.Destroy()

def main():
    image = cv2.VideoCapture(0)
    if not (image.isOpened()):
            image.open()

    image.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    image.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

    app = wx.App()
    layout = MainLayout(image)
    layout.init_algorithm()
    layout.Show(True)
    app.MainLoop()


if __name__ == '__main__':
    main()




