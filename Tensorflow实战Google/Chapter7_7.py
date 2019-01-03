
import tensorflow as tf
import matplotlib.pyplot as plt

image_raw_data = tf.gfile.FastGFile("/Users/zhangyiming/Desktop/TensorFlow_Google_DeepLearning_2/Data/Flower/flower_photos/dandelion/8181477_8cb77d2e0f_n.jpg", 'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)

    img_data = tf.image.convert_image_dtype(img_data,dtype=tf.float32)

    img_data = tf.image.resize_images(img_data,[180,267],method=1)

    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data,tf.float32),0)

    boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])

    result = tf.image.draw_bounding_boxes(batched,boxes)

    plt.imshow(result.eval())
    plt.show()