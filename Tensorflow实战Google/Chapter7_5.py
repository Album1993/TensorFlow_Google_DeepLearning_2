
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("/Users/zhangyiming/Desktop/TensorFlow_Google_DeepLearning_2/Data/Flower/flower_photos/dandelion/8181477_8cb77d2e0f_n.jpg", 'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)

    print(img_data)

    img_data = tf.image.convert_image_dtype(img_data,dtype=tf.float32)

    croped = tf.image.resize_image_with_crop_or_pad(img_data,300,200)

    padded = tf.image.resize_image_with_crop_or_pad(img_data,500,500)

    croped_img = tf.image.convert_image_dtype(croped,dtype=tf.uint8)

    padded_img = tf.image.convert_image_dtype(padded,dtype=tf.uint8)

    encoded_image = tf.image.encode_jpeg(croped_img)
    with tf.gfile.GFile('./Result/Chapter7_5_1.jpg','wb') as f:
        f.write(encoded_image.eval())

    encoded_image = tf.image.encode_jpeg(padded_img)
    with tf.gfile.GFile('./Result/Chapter7_5_2.jpg', 'wb') as f:
        f.write(encoded_image.eval())