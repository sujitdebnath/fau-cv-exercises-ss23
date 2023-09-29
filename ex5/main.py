from opticalFlowLK import OpticalFlowLK
from util import *
import numpy as np
import cv2


def test():
    ## Test for bilinear weights
    p = np.array([0.125, 0.82656])
    weights = np.array([0.15176, 0.02168, 0.72324, 0.10332])

    err = computeBilinerWeights(p) - weights
    e = np.linalg.norm(err)

    print("computeBilinearWeights error: " + str(e))
    if e < 1e-6:
        print("Test: SUCCESS!")
    else:
        print("Test: FAIL")

    ## Test for Gaussian kernel
    kernel = np.array(
        [[0.1690133273455962, 0.3291930023422986, 0.4111123050281957, 0.3291930023422986, 0.1690133273455962],
         [0.3291930023422986, 0.6411803997536073, 0.8007374099875735, 0.6411803997536073, 0.3291930023422986],
         [0.4111123050281957, 0.8007374099875735, 1, 0.8007374099875735, 0.4111123050281957],
         [0.3291930023422986, 0.6411803997536073, 0.8007374099875735, 0.6411803997536073, 0.3291930023422986],
         [0.1690133273455962, 0.3291930023422986, 0.4111123050281957, 0.3291930023422986, 0.1690133273455962]])
    res = computeGaussianWeights((5, 5), 0.3)
    err = res - kernel
    e = np.linalg.norm(err)
    print("computeGaussianWeights error: " + str(e))
    if e < 1e-6:
        print("Test: SUCCESS!")
    else:
        print("Reference " + str(kernel))
        print("Your result " + str(res))
        print("Test: FAIL")

    ## Tests for matrix inversion and gaussian weights
    A = np.array([[12, 4], [4, 8]])
    Ainv = np.array([[0.1, -0.05], [-0.05, 0.15]])
    err = invertMatrix2x2(A) - Ainv
    e = np.linalg.norm(err)

    print("invertMatrix2x2 error: " + str(e))
    if e < 1e-10:
        print("Test: SUCCESS!")
    else:
        print("Test: FAIL")

    print("=" * 30)


def main():
    test()

    cap = cv2.VideoCapture('../slow_traffic_small.mp4')

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    last_grey = None
    last_frame = None
    last_keypoints = []
    status = []

    winsize = [25, 25]

    while (cap.isOpened()):
        ret, frame = cap.read()
        scale = 0.7
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        termcrit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.03)
        subPixWinSize = (10, 10)

        keypoints = cv2.goodFeaturesToTrack(grey, 200, 0.01, 10, None, 3, False, 0.04)
        cv2.cornerSubPix(grey, keypoints, subPixWinSize, tuple([-1, -1]), termcrit)

        keypoints = np.array(keypoints)
        keypoints = np.squeeze(keypoints, axis=1)

        # keypoints = np.array([[302.279, 100.72]])

        flowPoints = 0
        if not len(last_keypoints) == 0:

            of = OpticalFlowLK(winsize, 0.03, 20)
            points, status = of.compute(last_grey, grey, np.copy(last_keypoints))

            for i in range(len(points)):

                if not status[i]:
                    continue

                diff = points[i] - last_keypoints[i]
                distance = np.linalg.norm(diff)

                if distance > 15 or distance < 0.2:
                    continue

                otherP = last_keypoints[i] + diff * 15
                flowPoints += 1

                color = tuple([0, 255, 0])
                cv2.circle(last_frame, tuple(last_keypoints[i]), 1, color)
                cv2.line(last_frame, tuple(last_keypoints[i]), tuple(otherP), color)

            cv2.imshow("out", last_frame)
            cv2.waitKey(1)

            print("[Keypoints] moving/total: {} / {}".format(flowPoints, len(points)))
        last_keypoints = np.copy(keypoints)
        last_grey = np.copy(grey)
        last_frame = np.copy(frame)


if __name__ == '__main__':
    main()