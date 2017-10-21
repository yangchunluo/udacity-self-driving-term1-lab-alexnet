import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet

nb_classes = 43
epochs = 3
batch_size = 128

# TODO: Load traffic signs data.
with open('./train.p', 'rb') as f:
    data = pickle.load(f)

# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(data['features'], data['labels'],
                                                      test_size=0.33, random_state=0)

# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(features, (227, 227))
one_hot_labels = tf.one_hot(labels, nb_classes)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc8W = tf.Variable(tf.truncated_normal(shape, 0, 1.0))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
tf.summary.scalar('loss', loss_operation)
optimizer = tf.train.AdamOptimizer(0.001)
training_operation = optimizer.minimize(loss_operation)

# TODO: Train and evaluate the feature extraction model.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
merged = tf.summary.merge_all()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={features: batch_x, labels: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./logs', sess.graph)

    print("Training...\n")
    batch_step = 0
    for i in range(epochs):
        print("EPOCH {} ...".format(i + 1))
        X_shuffled, y_shuffled = shuffle(X_train, y_train)
        for offset in range(0, len(X_train), batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_shuffled[offset:end], y_shuffled[offset:end]
            _, loss, summary = sess.run([training_operation, loss_operation, merged],
                                        feed_dict={features: batch_x, labels: batch_y})
            batch_step += 1
            writer.add_summary(summary, batch_step)
            if offset % (batch_size * 2) == 0:
                print("offset %d: loss=%f" % (offset, loss))

        print("Validation Accuracy = {:.3f}".format(evaluate(X_valid, y_valid)))
        # print("Training   Accuracy = {:.3f}".format(evaluate(X_train, y_train)))
        saver.save(sess, './alexnet-chkpt')
        print("Model checkpointed")
        print()
