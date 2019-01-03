
import tensorflow as tf
import matplotlib.pyplot as plt

image_raw_data = tf.gfile.FastGFile("/Users/zhangyiming/Desktop/TensorFlow_Google_DeepLearning_2/Data/Flower/flower_photos/dandelion/8181477_8cb77d2e0f_n.jpg", 'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)

    print(img_data)

    img_data = tf.image.convert_image_dtype(img_data,dtype=tf.float32)

    adjusted = tf.image.adjust_brightness(img_data,-0.5)

    adjusted = tf.clip_by_value(adjusted,0.0,1.0)

    plt.imshow(adjusted.eval())
    # plt.show()

    adjusted = tf.image.adjust_brightness(img_data,0.5)

    adjusted = tf.clip_by_value(adjusted,0.0,1.0)

    plt.imshow(adjusted.eval())
    plt.show()

    adjusted = tf.image.random_brightness(img_data,0.4)

    plt.imshow(adjusted.eval())
    # plt.show()



