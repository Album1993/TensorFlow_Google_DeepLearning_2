import matplotlib.pyplot as plt

import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("/Users/zhangyiming/Desktop/TensorFlow_Google_DeepLearning_2/Data/Flower/flower_photos/dandelion/8181477_8cb77d2e0f_n.jpg", 'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)

    print(img_data)

    img_data = tf.image.convert_image_dtype(img_data,dtype=tf.float32)

    resized = tf.image.resize_images(img_data,[300,300],method=0)

    resized_img = tf.image.convert_image_dtype(resized,dtype=tf.uint8)

    encoded_image = tf.image.encode_jpeg(resized_img)
    with tf.gfile.GFile('./Result/Chapter7_4.jpg','wb') as f:
        f.write(encoded_image.eval())