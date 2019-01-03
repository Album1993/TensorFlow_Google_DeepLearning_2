import matplotlib.pyplot as plt

import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("/Users/zhangyiming/Desktop/TensorFlow_Google_DeepLearning_2/Data/Flower/flower_photos/dandelion/8181477_8cb77d2e0f_n.jpg", 'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)

    print(img_data.eval())

    plt.imshow(img_data.eval())
    plt.show()
    print('showed')

    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile('./Result/Chapter7_3.jpg','wb') as f:
        f.write(encoded_image.eval())