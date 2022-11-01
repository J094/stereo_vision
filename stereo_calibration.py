'''
Author: J094
Date: 2022-01-26 15:04:30
LastEditTime: 2022-03-07 16:12:40
LastEditors: Please set LastEditors
Description: camera calibration
'''
import numpy as np
import cv2
import json

from pathlib import Path
import argparse
import logging

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
# create console handler and set level to info
ch = logging.StreamHandler()
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class StereoCalibrator(object):
    def __init__(self, imgPath: str, savePath: str, imgSize: tuple) -> None:
        self.imgPath = Path(imgPath)
        self.savePath = Path(savePath)
        # criteria for termination
        self.cameraCriteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.stereoCriteria = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        self.imgSize = imgSize
        self.cameraDict = dict()

    def _saveJSON(self):
        saveFile = self.savePath / 'stereo_parameters_1280x480.json'
        with open(saveFile, 'w') as wfile:
            json.dump(self.cameraDict, wfile, cls=NumpyEncoder4JSON)

    def _stereoCalibration(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0)
        objp = np.zeros((11*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        w, h = self.imgSize

        # Arrays to store object points and image points from all the images.
        objPoints = [] # 3d point in real world space
        imgPointsL = [] # 2d points in image plane left
        imgPointsR = [] # 2d points in image plane right

        logger.info("start to calibrate cameras...")

        for f in self.imgPath.glob('*.jpg'):
            s = f.stem
            frame = cv2.imread(str(f)) 
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayL = gray[:, :w]
            imgL = frame[:, :w]
            grayR = gray[:, w:]
            imgR = frame[:, w:]
            cv2.imwrite("./images_1280x480_left/" + s + ".jpg", imgL)
            cv2.imwrite("./images_1280x480_right/" + s + ".jpg", imgR)
            # cv2.imshow('imgL', imgL)
            # cv2.imshow('imgR', imgR)
            retL, cornersL = cv2.findChessboardCorners(grayL, (11,8), None)
            retR, cornersR = cv2.findChessboardCorners(grayR, (11,8), None)
            # If found, add object points, image points (after refining them)
            if retL == True and retR == True:
                objPoints.append(objp)

                corners2 = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), self.cameraCriteria)
                imgPointsL.append(cornersL)
                # Draw and display the corners
                cv2.drawChessboardCorners(imgL, (11,8), corners2, retL)
                cv2.imshow(f'calL{s}', imgL)
                cv2.waitKey(1000)

                corners2 = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), self.cameraCriteria)
                imgPointsR.append(cornersR)
                # Draw and display the corners
                cv2.drawChessboardCorners(imgR, (11,8), corners2, retR)
                cv2.imshow(f'calR{s}', imgR)
                cv2.waitKey(1000)

                cv2.waitKey(2000)
        cv2.destroyAllWindows()

        logger.info(f"num of objPoints: {len(objPoints)}")

        # refine camera matrix
        ret, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objPoints, imgPointsL, (w,h), None, None)
        ret, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objPoints, imgPointsR, (w,h), None, None)
        # reprojection error
        mean_errorL = 0
        for i in range(len(objPoints)):
            imgPoints2, _ = cv2.projectPoints(objPoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
            error = cv2.norm(imgPointsL[i], imgPoints2, cv2.NORM_L2)/len(imgPoints2)
            mean_errorL += error
        logger.info( "total error of left camera: {}".format(mean_errorL/len(objPoints)))
        mean_errorR = 0
        for i in range(len(objPoints)):
            imgPoints2, _ = cv2.projectPoints(objPoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
            error = cv2.norm(imgPointsR[i], imgPoints2, cv2.NORM_L2)/len(imgPoints2)
            mean_errorR += error
        logger.info( "total error of right camera: {}".format(mean_errorR/len(objPoints)))

        ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
            objPoints, imgPointsL, imgPointsR, mtxL, distL, mtxR, distR, 
            self.imgSize, criteria=self.stereoCriteria, flags=cv2.CALIB_FIX_INTRINSIC)

        # reprojection error
        mean_errorL = 0
        for i in range(len(objPoints)):
            imgPoints2, _ = cv2.projectPoints(objPoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
            error = cv2.norm(imgPointsL[i], imgPoints2, cv2.NORM_L2)/len(imgPoints2)
            mean_errorL += error
        logger.info( "total error of left camera: {}".format(mean_errorL/len(objPoints)))
        mean_errorR = 0
        for i in range(len(objPoints)):
            imgPoints2, _ = cv2.projectPoints(objPoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
            error = cv2.norm(imgPointsR[i], imgPoints2, cv2.NORM_L2)/len(imgPoints2)
            mean_errorR += error
        logger.info( "total error of right camera: {}".format(mean_errorR/len(objPoints)))

        self.cameraDict['imgSize'] = self.imgSize
        self.cameraDict['mtxL'] = mtxL
        self.cameraDict['distL'] = distL
        self.cameraDict['mtxR'] = mtxR
        self.cameraDict['distR'] = distR
        self.cameraDict['R'] = R
        self.cameraDict['T'] = T
        self.cameraDict['E'] = E
        self.cameraDict['F'] = F

        logger.info("cameraCalibration finished!")
    
    def stereoRectification(self):
        self._stereoCalibration()

        logger.info("start rectify cameras...")

        R1, R2, P1, P2, Q, roi1, roi2= cv2.stereoRectify(
            self.cameraDict['mtxL'], self.cameraDict['distL'], 
            self.cameraDict['mtxR'], self.cameraDict['distR'], 
            self.imgSize, self.cameraDict['R'], self.cameraDict['T'], alpha=1,
            )

        # mapLx, mapLy = cv2.initUndistortRectifyMap(self.cameraDict['mtxL'], self.cameraDict['distL'], R1, P1, self.imgSize, cv2.CV_32FC1)
        # mapRx, mapRy = cv2.initUndistortRectifyMap(self.cameraDict['mtxR'], self.cameraDict['distR'], R2, P2, self.imgSize, cv2.CV_32FC1)
        
        # self.cameraDict['mapLx'] = mapLx
        # self.cameraDict['mapLy'] = mapLy
        # self.cameraDict['mapRx'] = mapRx
        # self.cameraDict['mapRy'] = mapRy
        self.cameraDict['R1'] = R1
        self.cameraDict['R2'] = R2
        self.cameraDict['P1'] = P1
        self.cameraDict['P2'] = P2
        self.cameraDict['Q'] = Q
        self.cameraDict['roi1'] = roi1
        self.cameraDict['roi2'] = roi2
        logger.info(roi1)
        logger.info(roi2)
        
        self._saveJSON()

        logger.info("cameraRectification finished!")


class NumpyEncoder4JSON(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgPath', type=str, default='./images_1280x480')
    parser.add_argument('--imgSize', type=str, default='(640, 480)')
    parser.add_argument('--savePath', type=str, default='./')
    args = parser.parse_args()
    args.imgSize = eval(args.imgSize)
    calibrator = StereoCalibrator(imgPath=args.imgPath, savePath=args.savePath, imgSize=args.imgSize)
    calibrator.stereoRectification()
