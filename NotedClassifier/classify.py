import cPickle
import cv2
from hog import HOG
import os



model = open("svc1.cpickle").read()
model = cPickle.loads(model)



pathA2 = os.getcwd()+'/test/train_41_00031.jpg'
#print os.getcwd()
image = cv2.imread(pathA2)
print "#################################################"


#print image


hog = HOG(orientations=18,pixelsPerCell=(10, 10), cellsPerBlock=(1, 1))

#cv2.imshow("Eged",image)
#cv2.waitKey(0)

image = hog.clean(image)
histogram = hog.describe(image)
digit = model.predict(histogram)[0]
print digit
#cv2.imshow("Eged",image)
#scv2.waitKey(0)
