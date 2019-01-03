import tensorflow as tf

word_labels = tf.constant([2,0])

predict_logits = tf.constant([[2.0,-1.0,3.0],[1.0,0.0,-0.5]])


loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=word_labels,logits=predict_logits)

sess = tf.Session()

print(sess.run(loss))

word_prob_distribution = tf.constant([[0.0,0.0,1.0],[1.0,0.0,0.0]])

loss = tf.nn.softmax_cross_entropy_with_logits(labels=word_prob_distribution,logits=predict_logits)

print(sess.run(loss))

word_prob_smooth = tf.constant([[0.01,0.01,0.98],[0.98,0.01,0.01]])

loss = tf.nn.softmax_cross_entropy_with_logits(labels=word_prob_smooth,logits=predict_logits)

print(sess.run(loss))

sess.close()