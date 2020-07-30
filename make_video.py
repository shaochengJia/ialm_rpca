import numpy as np
import cv2

img_root = './ShoppingMall/'
fps = 30  # 保存视频的FPS，可以适当调整
size = (320, 256)
# 可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('./A_test.avi', fourcc, fps, size)

for i in range(0, 128):
    imgpath = './A/bgr/A_bgr_{}.png'.format(i)
    print(imgpath)
    frame = cv2.imread(imgpath)
    videoWriter.write(frame)

videoWriter.release()

videoWriter = cv2.VideoWriter('./E_test.avi', fourcc, fps, size)
for i in range(0, 128):
    imgpath = './E/bgr/E_bgr_{}.png'.format(i)
    print(imgpath)
    frame = cv2.imread(imgpath)
    videoWriter.write(frame)

videoWriter.release()
cv2.destroyAllWindows()