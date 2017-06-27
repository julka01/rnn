import glob
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def extract_features(parent_dir,sub_dirs,file_ext="*.wav",bands = 20, frames = 41):
    window_size = 512 * (frames - 1)
    mfccs = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip,s = librosa.load(fn)
            if fn.split('/')[2].split('_')[2]=='babycry':
                label=1
            elif fn.split('/')[2].split('_')[2]=='glassbreak':
                label=2
            elif fn.split('/')[2].split('_')[2]=='gunshot':
                label=3
            for (start,end) in windows(sound_clip,window_size):
                if(len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                    mfccs.append(mfcc)
                    labels.append(label)         
    features = np.asarray(mfccs).reshape(len(mfccs),bands,frames)
    return np.array(features), np.array(labels,dtype = np.int)

def one_hot_encode(labels):
    one_hot= pd.get_dummies(labels).values
    return one_hot


# use this to process the audio files into numpy arrays
def save_folds(data_dir):
    for k in range(1,2):
        fold_name = 'fold' + str(k)
        print "\nSaving " + fold_name
        features, labels = extract_features(parent_dir, [fold_name])
        labels = one_hot_encode(labels)
        
        print "Features of", fold_name , " = ", features.shape
        print "Labels of", fold_name , " = ", labels.shape
        
        feature_file = os.path.join(data_dir, fold_name + '_x.npy')
        labels_file = os.path.join(data_dir, fold_name + '_y.npy')
        np.save(feature_file, features)
        print "Saved " + feature_file
        np.save(labels_file, labels)
        print "Saved " + labels_file

def assure_path_exists(path):
    mydir = os.path.join(os.getcwd(), path)
    if not os.path.exists(mydir):
        os.makedirs(mydir)
        
# uncomment this to recreate and save the feature vectors


#print features, labels
parent_dir = 'data'     
save_dir = 'features'
assure_path_exists(save_dir)
#save_folds(save_dir)

    
# this is used to load the folds incrementally
def load_folds(folds):
    subsequent_fold = False
    for k in range(len(folds)):
        fold_name = 'fold' + str(folds[k])
        feature_file = os.path.join(data_dir, fold_name + '_x.npy')
        labels_file = os.path.join(data_dir, fold_name + '_y.npy')
        loaded_features = np.load(feature_file)
        loaded_labels = np.load(labels_file)
        print fold_name, "features: ", loaded_features.shape

        if subsequent_fold:
            features = np.concatenate((features, loaded_features))
            labels = np.concatenate((labels, loaded_labels))
        else:
            features = loaded_features
            labels = loaded_labels
            subsequent_fold = True
        
    return features, labels




data_dir = "features"
features,labels = load_folds([1])

#randomize and split data in train-test

rnd_indices = np.random.rand(len(labels)) < 0.90

train_x = features[rnd_indices]
train_y = labels[rnd_indices]
test_x = features[~rnd_indices]
test_y = labels[~rnd_indices]


tf.reset_default_graph()

learning_rate = 0.01
training_iters = 1000
batch_size = 50
display_step = 200

# Network Parameters
n_input = 20 
n_steps = 41
n_hidden = 300
n_classes = 3

x = tf.placeholder("float", [None, n_input, n_steps])
y = tf.placeholder("float", [None, n_classes])

weight = tf.Variable(tf.random_normal([n_hidden, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

def RNN(x, weight, bias):
    cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple = True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * 2)
    output, state = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    return tf.nn.softmax(tf.matmul(last, weight) + bias)


prediction = RNN(x, weight, bias)

# Define loss and optimizer
loss_f = -tf.reduce_sum(y * tf.log(prediction))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_f)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    
    for itr in range(training_iters):    
        offset = (itr * batch_size) % (train_y.shape[0] - batch_size)
        batch_x = train_x[offset:(offset + batch_size), :, :]
        batch_y = train_y[offset:(offset + batch_size), :]
        _, c = session.run([optimizer, loss_f],feed_dict={x: batch_x, y : batch_y})
            
        if itr % display_step == 0:
            # Calculate batch accuracy
            acc = session.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})
            print "Iter " + str(itr) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
    
    print('Test accuracy: ',round(session.run(accuracy, feed_dict={x: test_x, y: test_y}) , 3))
