from caffe2tf.network import Network
import tensorflow as tf
import cv2
# Create network.
import numpy as np
# import caffe
# Load variables (requires active Tensorflow session)
# sess = tf.Session()
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.

input_tensor = tf.placeholder(dtype=tf.float32, shape=[1,160,160,3], name="input_tensor")
image = cv2.imread('./01.jpg')

mean = np.array([_B_MEAN, _G_MEAN,  _R_MEAN], dtype=image.dtype)
ori_image = cv2.resize(image, dsize=(160,160)) 
whi = ori_image - [104, 117, 123]
w,h= 160,160
whi = np.array(whi, dtype='float32')
whi = whi[np.newaxis, :, :, :]
whi = tf.convert_to_tensor(whi)
net = Network("deploy-0508.prototxt", {'data': whi})

with tf.Session() as Sess:
    Sess.run(tf.global_variables_initializer())
    net.load("mynet.npy", Sess)
    s,b= net.layer_output('detection_out')
    face = b.eval()
    # print(face)
    face = face.reshape(100, 4)
    print(face)
    for loc in face:
        print(loc[1] * 160)
        cv2.rectangle(ori_image, (int(loc[0]*w), int(loc[1]*h)), (int(loc[2]*w), int(loc[3]*h)), (0, 0, 255), 3)
    cv2.imwrite('test2.jpg', ori_image)
    # output_val = Sess.run(output, feed_dict={
    #     input_tensor: [image]
    # })
    # print(output_val)
    
