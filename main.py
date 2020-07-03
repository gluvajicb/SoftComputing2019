import numpy as np
import cv2
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import sys
import imutils


'''
    Fx represents equation of a line for two points.

    Y - Y1

    Border line is at value 250 on Y scale.
    If the result of subtraction of Y values (Pedestrian - Line(Y) Value) is below 1 > Pedestrian crossed the borderline.
'''
def cross_detection(y):

    fx = y - 250

    if abs(fx) <= 1:
        return True

    return False


def process_video(video_path):

    video = cv2.VideoCapture(video_path)
    initFrame = None

    people_counter = 0

    while video:

        ret, frame = video.read()

        if frame is False:
            print("Can not load video frame!")

        if ret is False:
            break

        frame = imutils.resize(frame, 800)
        # Converting RGB to GRAY_IMAGE
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blurring sharp lines in image
        gray_image = cv2.GaussianBlur(gray_image, (19, 19), 1)

        '''
            Uzima se prvi frame u video-u i kasnije se poredi sa frame-ovima koji slede u video-u.
        '''
        if initFrame is None:
            initFrame = gray_image
            continue


        # Calculates difference in pixelization between initial frame and current frame.
        difference = cv2.absdiff(initFrame, gray_image)

        # Calculates Threshold for Difference
        threshold = cv2.threshold(difference, 20, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold, None, iterations=2)

        contours = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

        for contour in contours:

            if cv2.contourArea(contour) < 90:
                continue

            '''
                Width represents width of a rectangle.
                Height represents height of a rectangle.

                Point(x, y) is the bottom left point of a rectangle.
            '''
            (x, y, width, height) = cv2.boundingRect(contour)
            # Bottom Left Point of a rectangle.
            bottomLeftPoint = (x, y)
            # Upper Right Point of a rectangle.
            upperRightPoint = (x + width, y + height)

            # Draws a rectangle.
            '''
                First arg> Frame
                Second arg> Bottom left point of a rectangle.
                Third arg> Upper Right Point of a rectangle.
                Fourth arg> Color of the borders of a rectangle.
                Fifth arg> Thickness of the border.
            '''
            cv2.rectangle(frame, bottomLeftPoint, upperRightPoint, (0, 0, 255), 1)

            # Calculating Center Point on X scale and Y scale.
            rectangleCenterX = (x + x + width) / 2
            rectangleCenterY = (y + y + height) / 2

            '''
                Draws a line in the video.
            '''
            cv2.line(frame, (0, 250), (800, 250), (0, 0, 255), 3)

            if (cross_detection(rectangleCenterY)):
                people_counter += 1

        '''
            Enables playing a video.
        '''
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        '''
            Parameters:
                image, text, coordinates of the bottom-left corner of the text string in the image,
                font, fontScale, color, thickness, lineType, BottomLeftOrigin, ReturnValue[image]
        '''
        cv2.putText(frame, "People counted: {}".format(str(people_counter)), (20, 50),
                    cv2.QT_FONT_NORMAL, 1, (0, 255, 255), 1)
        cv2.imshow("Soft Computing 2019", frame)

    print(people_counter)

    video.release()
    cv2.destroyAllWindows()

    return people_counter



if __name__ == '__main__':

    # Processing all videos. Returns People Counter for each video.
    pc1 = process_video("video_data/video1.mp4")
    pc2 = process_video("video_data/video2.mp4")
    pc3 = process_video("video_data/video3.mp4")
    pc4 = process_video("video_data/video4.mp4")
    pc5 = process_video("video_data/video5.mp4")
    pc6 = process_video("video_data/video6.mp4")
    pc7 = process_video("video_data/video7.mp4")
    pc8 = process_video("video_data/video8.mp4")
    pc9 = process_video("video_data/video9.mp4")
    pc10 = process_video("video_data/video10.mp4")

    y = [4, 24, 17, 23, 17, 27, 29, 22, 10, 23]
    y_predicted = [pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8, pc9, pc10]

    mae = mean_absolute_error(y, y_predicted)
    print("Mean Absolute Error is: ", mae)





