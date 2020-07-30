from scipy import misc
import numpy as np
from glob import glob
from inexact_augmented_lagrange_multiplier import inexact_augmented_lagrange_multiplier
from scipy.io import savemat
import os
import sys
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from matplotlib import cm
import matplotlib
import imageio
import cv2

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray / 255


def make_video(alg, num, cache_path='./matrix_IALM_tmp'):
    name = alg
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    # If you generate a big
    if not os.path.exists('%s/%s_tmp' % (cache_path, name)):
        os.mkdir("%s/%s_tmp" % (cache_path, name))
    mat = loadmat('./%s_background_subtraction.mat' % (name))
    org = X.reshape(d1, d2, num) * 255.
    # print('d1 {}, d2 {}'.format(d1, d2))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    usable = [x for x in sorted(mat.keys()) if "_" not in x][0]
    sz = min(org.shape[2], mat[usable].shape[2])
    for i in range(num):
        ax.cla()
        ax.axis("off")
        ax.imshow(np.hstack([mat[x][:, :, i] for x in sorted(mat.keys()) if "_" not in x] + \
                            [org[:, :, i]]), cm.gray)
        fname_ = '%s/%s_tmp/_tmp%03d.png' % (cache_path, name, i)
        if (i % 25) == 0:
            print('Completed frame', i, 'of', sz, 'for method', name)
        fig.tight_layout()
        fig.savefig(fname_, bbox_inches="tight")
    plt.close()


def splitFB(X, d1, d2, num, channel, model_name):
    X = X.reshape(d1 * d2, num)
    A, E = inexact_augmented_lagrange_multiplier(X[:, :])
    A = A.reshape(d1, d2, num) * 255.
    E = E.reshape(d1, d2, num) * 255.
    A_save_path = './A/{}/'.format(channel)
    E_save_path = './E/{}/'.format(channel)
    if not os.path.exists(A_save_path):
        os.mkdir(A_save_path)
    if not os.path.exists(E_save_path):
        os.mkdir(E_save_path)
    for i in range(num):
        tA, tE = A[:, :, i], E[:, :, i]
        cv2.imwrite(A_save_path + 'A_{}.png'.format(i), tA)
        cv2.imwrite(E_save_path + 'E_{}.png'.format(i), tE)

    # savemat("./{}_{}_IALM_background_subtraction.mat".format(model_name, channel), {"1": A, "2": E})
    print('{} model saved.'.format(channel))
    # make_video(alg='{}_{}_IALM'.format(model_name, channel), num=num)


if __name__ == "__main__":
    names = sorted(glob("./ShoppingMall/*.bmp"))
    d1, d2, channels = imageio.imread(names[0]).shape  # (256, 320, 3)
    # d1 = 128
    # d2 = 160
    num = len(names)
    # i = names[0]
    # frame = misc.imread(i).astype(np.double)
    # print(type(frame))
    # print(frame.shape)
    # bX = frame[:, :, 0] / 255.
    # gX = frame[:, :, 1] / 255.
    # rX = frame[:, :, 2] / 255.
    # cv2.imshow('bX', bX)
    # cv2.imshow('gX', gX)
    # cv2.imshow('rX', rX)
    # cv2.waitKey(0)

    X = np.zeros((d1, d2, num))
    rX = np.zeros((d1, d2, num))
    gX = np.zeros((d1, d2, num))
    bX = np.zeros((d1, d2, num))
    for n, i in enumerate(names):
        tmp_frame = misc.imread(i).astype(np.double)
        bX[:, :, n] = tmp_frame[:, :, 0] / 255.
        gX[:, :, n] = tmp_frame[:, :, 1] / 255.
        rX[:, :, n] = tmp_frame[:, :, 2] / 255.
        # X[:, :, n] = rgb2gray(misc.imread(i).astype(np.double)) / 255.
    bX = bX.reshape(d1 * d2, num)
    gX = gX.reshape(d1 * d2, num)
    rX = rX.reshape(d1 * d2, num)
    print(bX[:, :].shape)
    model_name = 't1'
    splitFB(bX[:, :], d1, d2, num, 'b', model_name)
    splitFB(gX[:, :], d1, d2, num, 'g', model_name)
    splitFB(rX[:, :], d1, d2, num, 'r', model_name)