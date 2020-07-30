import cv2
import os

# A
b_path = './A/b/'
g_path = './A/g/'
r_path = './A/r/'
bgr_path = './A/bgr/'

for i in range(0, 128):
    if not os.path.exists(b_path):
        os.mkdir(b_path)
    if not os.path.exists(g_path):
        os.mkdir(g_path)
    if not os.path.exists(r_path):
        os.mkdir(r_path)
    if not os.path.exists(bgr_path):
        os.mkdir(bgr_path)
    b = cv2.imread(b_path + 'A_{}.png'.format(i))
    g = cv2.imread(g_path + 'A_{}.png'.format(i))
    r = cv2.imread(r_path + 'A_{}.png'.format(i))
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    merged = cv2.merge([r, g, b])
    cv2.imwrite(bgr_path + 'A_bgr_{}.png'.format(i), merged)

# E
b_path = './E/b/'
g_path = './E/g/'
r_path = './E/r/'
bgr_path = './E/bgr/'

for i in range(0, 128):
    if not os.path.exists(b_path):
        os.mkdir(b_path)
    if not os.path.exists(g_path):
        os.mkdir(g_path)
    if not os.path.exists(r_path):
        os.mkdir(r_path)
    if not os.path.exists(bgr_path):
        os.mkdir(bgr_path)
    b = cv2.imread(b_path + 'E_{}.png'.format(i))
    g = cv2.imread(g_path + 'E_{}.png'.format(i))
    r = cv2.imread(r_path + 'E_{}.png'.format(i))
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    merged = cv2.merge([r, g, b])
    cv2.imwrite(bgr_path + 'E_bgr_{}.png'.format(i), merged)