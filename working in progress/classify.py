import cPickle
import cv2
from hog import HOG
import os


class Classify:

    def __init__(self):
        self.model = open("svc2.cpickle").read()
        self.model = cPickle.loads(model)
        self.hog = HOG(orientations=18, pixelsPerCell=(10, 10), cellsPerBlock=(1, 1))

    def Classify(self,image):
        image = hog.clean(self.image)
        histogram = hog.describe(image)
        digit = model.predict(histogram)[0]
        return digit


'''
model = open("svc2.cpickle").read()
model = cPickle.loads(model)


path = '/home/competitive/Downloads/English/Img/GoodImg/Bmp/Sample012/img012-00004.png'
#pathA2 = os.getcwd()+'/test/train_41_00031.jpg'
#print os.getcwd()
image = cv2.imread(path)
print "#################################################"


#print image


hog = HOG(orientations=18,pixelsPerCell=(10, 10), cellsPerBlock=(1, 1))

#cv2.imshow("Eged",image)
#cv2.waitKey(0)

image = hog.clean(image)
histogram = hog.describe(image)
digit = model.predict(histogram)[0]
print ("The prediction is "+digit)
#cv2.imshow("Eged",image)
#cv2.waitKey(0)
'''
