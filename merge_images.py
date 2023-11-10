#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:38:38 2023

@author: pranomvignesh
"""

import cv2
import os

frame_width = 480
frame_height = 640
fps = 3

images_directory = '/Users/pranomvignesh/WorkFolder/yubie/fetch/ball/colord2/'
folders = ['left', 'right']



for folder in folders:
    directory = f'{images_directory}/{folder}'
    image_files = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]
    image_files.sort()
    out = cv2.VideoWriter(f'{directory}/output_video_{folder}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        img = cv2.imread(image_path)
        resized_img = cv2.resize(img, (frame_width, frame_height))
        out.write(resized_img)

    out.release()
cv2.destroyAllWindows()
