import cv2
import cPickle as pickle
import glob
from shutil import copyfile


class Extract:
    def __init__(self):
        self.save_file = 'datasets/faces_training.pkl'
        self.emotions = [
                        "neutral",
                        "anger",
                        "contempt", "disgust",
                        "fearful",
                        "happy",
                        "sadness",
                        "surprise"]
        self.labels = []
        self.pictures = []

    # take pictures from dataset and copy them to corresponding folders with emotions labels
    def sortSet(self):
        folders = glob.glob("\\users\\matik\\Dataset\\source_emotion\\*")
        for f in folders:
            partName = "%s" % f[-4:]
            for sequences in glob.glob("%s\\*" % (f)):
                for files in glob.glob("%s\\*" % sequences):
                    current_session = sequences[-3:]
                    file = open(files, 'r')
                    emotion = int(float(file.readline()))
                    if emotion != 1 and emotion != 2 and emotion != 3 and emotion != 4:
                        for i in xrange(1, 6):
                            # get path for last image in sequence, which contains the emotion
                            srcfile_emotion = \
                                glob.glob("\\users\\matik\\Dataset\\source_images\\%s\\%s\\*" % (
                                        partName, current_session))[-i]
                            # Do same for emotion containing image
                            dest_emot = ("\\users\\matik\\Dataset\\sorted_set\\%s\\%s") % (
                                    self.emotions[emotion], srcfile_emotion[44:])
                            copyfile(srcfile_emotion, dest_emot)
                            if i < 3:
                                srcfile_neutral = \
                                glob.glob("\\users\\matik\\Dataset\\source_images\\%s\\%s\\*" % (
                                    partName, current_session))[i-1]
                                dest_neut = ("\\users\\matik\\Dataset\\sorted_set\\neutral\\%s" % srcfile_neutral[44:])
                                copyfile(srcfile_neutral, dest_neut)

    # extract faces from given dataset and put them into a list. Dump pictures to binary representation
    # and save as pickle file
    def extractFaces(self, emotion):
        face_cascade = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")
        images = glob.glob("\\users\\matik\\Dataset\\sorted_set\\%s\\*" %emotion)
        for image in images:
            picture=cv2.imread(image)
            grayPicture=cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
            face=face_cascade.detectMultiScale(grayPicture,1.3,5)
            if (len(face)==1):
                for (x,y,w,h) in face:
                    grayPicture=grayPicture[y:y + h, x:x + w]
                    try:
                        outputPic=cv2.resize(grayPicture, (250, 250))
                        self.labels.append(emotion)
                        self.pictures.append(outputPic.flatten())
                    except:
                        pass

    def main(self):
        # self.sortSet()
        for emotion in self.emotions:
            self.extractFaces(emotion)
        f = open(self.save_file, 'wb')
        pickle.dump(self.pictures, f)
        pickle.dump(self.labels, f)


if __name__ == '__main__':
    extract = Extract()
    extract.main()






