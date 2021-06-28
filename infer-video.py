from argparse import ArgumentParser
from tensorflow import keras
import numpy as np
import cv2
import os

parser = ArgumentParser()


def main():
    args = parser.parse_args()

    model = keras.models.load_model('models/generator.h5')
    inputs = keras.Input((None, None, 3))
    output = model(inputs)
    model = keras.models.Model(inputs, output)

    #for videos

    cap = cv2.VideoCapture('/home/ayush-ai/Music/talking_head/talking-head-anime-demo/input_video/3.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_vid = cv2.VideoWriter('./outputs/srgan-output.mp4', fourcc, 25, (1024,1024))
    while True: 
        there_is_frame, frame = cap.read()
        if not there_is_frame:
            return

    
        low_res = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        low_res = low_res / 255.0

        sr = model.predict(np.expand_dims(low_res, axis=0))[0]

        sr = ((sr + 1) / 2.) * 255

        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        # print(sr.shape)
        # print(sr.dtype)
        res = sr.astype(np.uint8)
        output_vid.write(res)

    output_vid.release()
    cap.release()

if __name__ == '__main__':
    main()
