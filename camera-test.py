import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

while not vc.isOpened():
    pass

rval, frame = vc.read()
frame = frame[30:-30:3, 190:-190:3, :]

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    frame = frame[30:-30:3, 190:-190:3, :]
    print(frame.shape)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")
