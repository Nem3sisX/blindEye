import cv2, sys, numpy, os
from gtts import gTTS
from pygame import mixer
import time

size = 4
datasets = "datasets"
print("Recognizing Face Please Be in sufficient Lights...")
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + "/" + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(width, height) = (130, 100)
(images, lables) = [numpy.array(lis) for lis in [images, lables]]
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
webcam = cv2.VideoCapture(1)
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y : y + h, x : x + w]
        face_resize = cv2.resize(face, (width, height))
        # Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1] < 80:
            cv2.putText(
                im,
                "% s - %.0f" % (names[prediction[0]], prediction[1]),
                (x - 10, y - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 0),
            )
            """         
            file1 = str("prediction.mp3")
            mytext = str(names[prediction[0]] + " is here.")
            language = "en"
            myobj = gTTS(text=mytext, lang=language, slow=False)
            myobj.save(file1)
            mixer.init()
            mixer.music.load(file1)
            mixer.music.play()"""
        else:
            cv2.putText(
                im,
                "not recognized",
                (x - 10, y - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 0),
            )
    cv2.imshow("OpenCV", im)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()

