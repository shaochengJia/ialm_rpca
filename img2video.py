import cv2

img_root = './ShoppingMall/'
fps = 30  # 保存视频的FPS，可以适当调整
size = (630, 182)
# 可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('./IALM_test.avi', fourcc, fps, size)  # 最后一个是保存图片的尺寸

# for i in range(1481, 1609):
#     imgpath = './ShoppingMall/ShoppingMall{}.bmp'.format(i)
#     # print(imgpath)
#     frame = cv2.imread(imgpath)
#     print(frame)
#     videoWriter.write(frame)

for i in range(0, 100):
    imgpath = './matrix_IALM_tmp/t2IALM_tmp/_tmp{}.png'.format('%03d'%(i))
    print(imgpath)
    frame = cv2.imread(imgpath)
    videoWriter.write(frame)

videoWriter.release()
cv2.destroyAllWindows()