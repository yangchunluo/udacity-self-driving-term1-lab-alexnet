from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import csv
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten

# Load training data.
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this it's a good idea to flatten the array.
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

image_shape = X_train[0].shape
n_classes = 10

# Split the training data for validation.
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                      test_size=0.3, random_state=42, stratify=y_train)
n_train = len(X_train)

# Load label to sign name dictionary.
reader = csv.reader(open('signnames.csv', 'r'))
signnames = {}
for row in reader:
    id, meaning = row
    if id == "ClassId":
        continue
    signnames[int(id)] = meaning

# Normalization.
def normalize(X):
    return (X.astype(np.float32) - 128) / 128

X_train_normalized = normalize(X_train)
X_valid_normalized = normalize(X_valid)
X_test_normalized = normalize(X_test)

# Model architecture.
def LuoNet(inputs, keep_prob):
    # Arguments used for tf.truncated_normal,
    # Randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1
    with tf.name_scope('layer1-conv'):
        # Convolution. Input = 32x32x3. Output = 28x28x6.
        conv1 = tf.nn.conv2d(inputs,
                             tf.Variable(tf.truncated_normal((5, 5, image_shape[2], 6), mu, sigma)),
                             [1, 1, 1, 1],
                             'VALID') + tf.Variable(tf.zeros(6))
        # Activation.
        conv1 = tf.nn.relu(conv1)
        # Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = tf.nn.max_pool(conv1,
                               [1, 2, 2, 1],
                               [1, 2, 2, 1],
                               'VALID')

    # Layer 2
    with tf.name_scope('layer2-conv'):
        # Convolutional. Output = 10x10x16.
        conv2 = tf.nn.conv2d(conv1,
                             tf.Variable(tf.truncated_normal((5, 5, 6, 16), mu, sigma)),
                             [1, 1, 1, 1],
                             'VALID') + tf.Variable(tf.zeros(16))
        # Activation.
        conv2 = tf.nn.relu(conv2)
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2,
                               [1, 2, 2, 1],
                               [1, 2, 2, 1],
                               'VALID')

    # Layer 3
    with tf.name_scope('layer3-fc'):
        # Flatten. Input = 5x5x16. Output = 400.
        fc0 = flatten(conv2)
        # Fully Connected. Input = 400. Output = 120.
        fc1 = tf.matmul(fc0,
                        tf.Variable(tf.truncated_normal((400, 120), mu, sigma))) + tf.Variable(tf.zeros(120))
        # Activation.
        fc1 = tf.nn.relu(fc1)

    # Layer 3a: dropout
    with tf.name_scope('layer3a-dropout'):
        fc1_drop = tf.nn.dropout(fc1, keep_prob)

    # Layer 4
    with tf.name_scope('layer4-fc'):
        # Fully Connected. Input = 120. Output = 84.
        fc2 = tf.matmul(fc1_drop,
                        tf.Variable(tf.truncated_normal((120, 84), mu, sigma))) + tf.Variable(tf.zeros(84))
        # Activation.
        fc2 = tf.nn.relu(fc2)

    # Layer 4a: dropout
    with tf.name_scope('layer4a-dropout'):
        fc2_drop = tf.nn.dropout(fc2, keep_prob)

    # Layer 5
    with tf.name_scope('layer5-fc'):
        # Fully Connected. Input = 84. Output = n_classes.
        logit = tf.matmul(fc2_drop, tf.Variable(tf.truncated_normal((84, n_classes), mu, sigma))) + tf.Variable(
            tf.zeros(n_classes))

    return logit


# Rest of the graph.
inputs = tf.placeholder(tf.float32, np.concatenate((np.array([None]), image_shape)))
labels = tf.placeholder(tf.int32, (None))
one_hot_labels = tf.one_hot(labels, n_classes)
keep_prob = tf.placeholder(tf.float32)

logits = LuoNet(inputs, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)

INIT_RATE = 0.001
BATCH_SIZE = 128

# Exponential delay of learning rate
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(INIT_RATE, global_step,
                                           n_train/BATCH_SIZE, 0.96, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)
training_operation = optimizer.minimize(loss_operation, global_step=global_step)

# Model accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={inputs: batch_x, labels: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


EPOCHS = 10
KEEP_PROB = 0.5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("Training...\n")
    for i in range(EPOCHS):
        print("EPOCH {} ...".format(i + 1))
        X_shuffled, y_shuffled = shuffle(X_train_normalized, y_train)
        for offset in range(0, n_train, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_shuffled[offset:end], y_shuffled[offset:end]
            _, loss, rate, step = sess.run([training_operation, loss_operation, learning_rate, global_step],
                                           feed_dict={inputs: batch_x, labels: batch_y, keep_prob: KEEP_PROB})
            if offset % (BATCH_SIZE * 10) == 0:
                print("offset %d: loss=%f, step=%d, rate=%f" % (offset, loss, step, rate))

        print("Validation Accuracy = {:.3f}".format(evaluate(X_valid_normalized, y_valid)))
        print("Training   Accuracy = {:.3f}".format(evaluate(X_train_normalized, y_train)))
        saver.save(sess, './lenet-cifra10-chkpt')
        print("Model checkpointed")
        print()

    print("Test Accuracy = {:.3f}".format(evaluate(X_test_normalized, y_test)))