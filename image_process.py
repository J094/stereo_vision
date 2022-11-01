'''
Author: J094
Date: 2022-01-27 16:21:35
LastEditTime: 2022-03-07 15:26:10
LastEditors: Please set LastEditors
Description: image process
'''

import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt

from pathlib import Path
import json
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

class imageProcessor(object):
    def __init__(self, readPath: str, imgSize=(640,480)) -> None:
        self.readPath = Path(readPath)
        self.cameraDict = self._readJSON()
        # logger.info(self.cameraDict['roi1'])
        # logger.info(self.cameraDict['roi2'])
        self.rectifyMaps = dict()
        # calculate and store RectifyMaps once
        self._getRectifyMaps()
        self.imgSize = imgSize

    def _readJSON(self):
        readFile = self.readPath / 'stereo_parameters_1280x480.json'
        with open(readFile, 'r') as rfile:
            cameraDict = json.load(rfile)
        # transform list to np.array
        for key in cameraDict:
            if isinstance(cameraDict[key], list):
                cameraDict[key] = np.asarray(cameraDict[key])
        return cameraDict
    
    def _getRectifyMaps(self):
        mapLx, mapLy = cv2.initUndistortRectifyMap(
            self.cameraDict['mtxL'], self.cameraDict['distL'], 
            self.cameraDict['R1'], self.cameraDict['P1'], 
            tuple(self.cameraDict['imgSize']), cv2.CV_32FC1,
            )
        mapRx, mapRy = cv2.initUndistortRectifyMap(
            self.cameraDict['mtxR'], self.cameraDict['distR'], 
            self.cameraDict['R2'], self.cameraDict['P2'], 
            tuple(self.cameraDict['imgSize']), cv2.CV_32FC1,
            )
        self.rectifyMaps['mapLx'] = mapLx
        self.rectifyMaps['mapRx'] = mapRx
        self.rectifyMaps['mapLy'] = mapLy
        self.rectifyMaps['mapRy'] = mapRy

    def preprocess(self, img):
        w, h = self.cameraDict['imgSize']
        # transform to grayscale image
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cut image to left and right parts
        imgL = img[:, :w]
        imgR = img[:, w:]
        # undistort and rectify images
        imgLRe = cv2.remap(
            imgL, self.rectifyMaps['mapLx'], self.rectifyMaps['mapLy'],
            interpolation=cv2.INTER_AREA,
            )
        imgRRe = cv2.remap(
            imgR, self.rectifyMaps['mapRx'], self.rectifyMaps['mapRy'],
            interpolation=cv2.INTER_AREA,
            )
        # crop image
        # xL, yL, wL, hL = self.cameraDict['roi1']
        # imgLRe = imgLRe[yL:yL+hL, xL:xL+wL]
        # xR, yR, wR, hR = self.cameraDict['roi2']
        # imgRRe = imgRRe[yR:yR+hR, xR:xR+wR]
        # resize image
        if w != self.imgSize[0] or h != self.imgSize[1]:
            imgLRe = cv2.resize(imgLRe, self.imgSize, interpolation=cv2.INTER_AREA)
            imgRRe = cv2.resize(imgRRe, self.imgSize, interpolation=cv2.INTER_AREA)
        return imgLRe, imgRRe

    # draw line to test rectified results
    def draw_line(self, image1, image2):
        # 建立输出图像
        height = max(image1.shape[0], image2.shape[0])
        width = image1.shape[1] + image2.shape[1]
        output = np.zeros((height, width), dtype=np.uint8)
        output[0:image1.shape[0], 0:image1.shape[1]] = image1
        output[0:image2.shape[0], image1.shape[1]:] = image2
        # 绘制等间距平行线
        line_interval = 50  # 直线间隔：50
        for k in range(height // line_interval):
            cv2.line(
                output, (0, line_interval * (k + 1)), (2 * width, 
                line_interval * (k + 1)), (0, 255, 0), thickness=2, 
                lineType=cv2.LINE_AA,
                )
        return output
    
    # calculate desparities
    def stereoMatchSGBM(self, left_image, right_image, down_scale=False):
        # SGBM匹配参数设置
        if left_image.ndim == 2:
            img_channels = 1
        else:
            img_channels = 3
        min_disp = 1
        num_disp = 48 - min_disp
        blockSize = 9
        paraml = {
            'minDisparity': min_disp,
            'numDisparities': num_disp,
            'blockSize': blockSize,
            'P1': 8 * img_channels * blockSize ** 2,
            'P2': 32 * img_channels * blockSize ** 2,
            'disp12MaxDiff': 3,
            'preFilterCap': 63,
            'uniquenessRatio': 5,
            'speckleWindowSize': 100,
            'speckleRange': 3,
            'mode': cv2.StereoSGBM_MODE_SGBM_3WAY,
            #'mode': cv2.StereoSGBM_MODE_HH4,
            }
        # 构建SGBM对象
        left_matcher = cv2.StereoSGBM_create(**paraml)
        # 计算视差图
        size = (left_image.shape[1], left_image.shape[0])
        if down_scale == False:
            disparity_left = left_matcher.compute(left_image, right_image)
        else:
            left_image_down = cv2.pyrDown(left_image)
            right_image_down = cv2.pyrDown(right_image)
            factor = left_image.shape[1] / left_image_down.shape[1]
            disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
            disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
            disparity_left = factor * disparity_left
        # 真实视差（因为SGBM算法得到的视差是×16的）
        disp_left = disparity_left.astype(np.float32) / 16.0
        coord_3d = cv2.reprojectImageTo3D(disp_left, self.cameraDict['Q'], handleMissingValues=True)
        return disp_left, coord_3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--readPath', type=str, default='./')
    args = parser.parse_args()
    imgProcessor = imageProcessor(readPath=args.readPath)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.warning("Cannot open camera")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        # capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            logger.warning("Can't receive frame (stream end?). Exiting ...")
            break
        imgL, imgR = imgProcessor.preprocess(frame)
        cv2.imshow('imgL', imgL)
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        # test = imgProcessor.draw_line(imgL, imgR)
        # cv2.imshow('test', test)
        dispL, coord3D = imgProcessor.stereoMatchSGBM(imgL, imgR, True)
        dispL = dispL.astype(np.uint8)
        #coord3D = coord3D.astype(np.uint8)
        heatmapL = cv2.applyColorMap(dispL, cv2.COLORMAP_HOT)
        heatmapL = heatmapL
        #heatmap3D = cv2.applyColorMap(coord3D, cv2.COLORMAP_HOT)
        #cv2.imshow('coord3D', coord3D)
        # print(np.shape(coord3D))
        #cv2.imshow('dispL', dispL)
        cv2.imshow('heatmapL', heatmapL)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
