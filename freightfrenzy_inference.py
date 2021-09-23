import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="assets/FreightFrenzy_BC.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# initialize camera
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
while not vc.isOpened():
    pass


def float2imgpt(x):
    return max(min(int(300 / x), 299), 0)


while True:
    # get image from camera
    rval, frame = vc.read()

    # shape to (300, 300, 3)
    frame = frame[30:-30:3, 190:-190:3, :]

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array([frame / 255], dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    # output tensors are 247-250
    rects = interpreter.get_tensor(247)
    print(rects)
    labels = interpreter.get_tensor(248)
    print(labels)
    confidences = interpreter.get_tensor(249)
    print(confidences)
    print('doot')
    for top_rect in rects[0]:
        # print(frame.shape)
        # print((float2imgpt(top_rect[0]), float2imgpt(top_rect[1])),
        #       (float2imgpt(top_rect[2]), float2imgpt(top_rect[3])), )
        frame = cv2.rectangle(frame.astype(np.uint8).copy(),
                              (float2imgpt(top_rect[0]), float2imgpt(top_rect[1])),
                              (float2imgpt(top_rect[2]), float2imgpt(top_rect[3])),
                              (0, 255, 0))
    cv2.imshow("preview", frame)
    cv2.waitKey(20)

vc.release()
cv2.destroyWindow("preview")
