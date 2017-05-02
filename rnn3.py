import tensorflow as tf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import arff
import random
import pandas as pd

#read dataset from file
dataset = arff.load(open('1sample-llds-is13-train.arff','rb'))
data = dataset['data']
random.shuffle(data)
labels = []

for i in data:
    i.remove(i[0])
    len1= len(i)
    labels.append(i[len1-1])
    i.remove(i[len1-1])
    
data= np.array(data)
labels=np.array(labels)
labels = pd.get_dummies(labels).values
labels= np.concatenate([labels ,np.zeros((len(labels),125))], 1)
# create train and test samples
#data = (data - data.min(0)) / x.ptp(0)                                                                  
train_index = int(0.8 * len(data))
x_train = data[:train_index]
x_test = data[train_index :]
y_train = labels[:train_index]
y_test = labels[train_index:]


# hyperparameters
                    
num_epochs = 50
total_series_length = len(x_train)
truncated_backprop_length = 131
state_size = 32
num_classes = 131
batch_size = 100
num_batches = int(total_series_length/batch_size)

#build the model

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, 131])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, 131])
#placeholder for rnn state
init_state = tf.placeholder(tf.float32, [batch_size, state_size])

# weights and biases

#randomly initialize weights
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
#anchor, improves convergance, matrix of 0s 
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
#Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
#so a bunch of arrays, 1 batch per time step
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

#Forward pass
#state placeholder
current_state = init_state
#series of states through time
states_series = []


#for each set of inputs
#forward pass through the network to get new state value
#store all states in memory
for current_input in inputs_series:
    #format input
    current_input = tf.reshape(current_input, [batch_size, 1])
    #mix both state and input data 
    input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns
    #perform matrix multiplication between weights and input, add bias
    #squash with a nonlinearity, for probabiolity value
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    #store the state in memory
    states_series.append(next_state)
    #set current state to next one
    current_state = next_state


#calculate loss
#second part of forward pass
#logits short for logistic transform
logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
#apply softmax nonlinearity for output probability
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

#measure loss, calculate softmax again on logits, then compute cross entropy
#measures the difference between two probability distributions
#this will return A Tensor of the same shape as labels and of the same type as logits 
#with the softmax cross entropy loss.
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits =logits_series , labels = labels_series)]
#computes average, one value
total_loss = tf.reduce_mean(losses)

#use adagrad to minimize with .3 learning rate
#minimize it with adagrad, not SGD
#weights that receive high gradients will have their effective learning rate reduced, 
#while weights that receive small or infrequent updates will have their effective learning rate increased. 
#reference http://seed.ucsd.edu/mediawiki/images/6/6a/Adagrad.pdf
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
correct = tf.equal(tf.argmax(predictions_series,1),tf.argmax(labels_series,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

'''
#visualizer
def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(batch_size):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)

'''
#Step 3 Training the network
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    # normalize the data
    x_train = sess.run(tf.nn.l2_normalize(x_train, 0, epsilon=1e-12, name=None))
    x_test = sess.run(tf.nn.l2_normalize(x_test, 0, epsilon=1e-12, name=None))
    #interactive mode
    #plt.ion()
    #initialize the figure
    #plt.figure()
    #show the graph
    #plt.show()
    #to show the loss decrease
    loss_list = []

    for epoch_idx in range(num_epochs):
        #generate data at eveery epoch, batches run in epochs
        
        #initialize an empty hidden state
        _current_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)
        #each batch
        for batch_idx in range(num_batches):
            #starting and ending point per batch
            #since weights reoccuer at every layer through time
            #These layers will not be unrolled to the beginning of time, 
            #that would be too computationally expensive, and are therefore truncated 
            #at a limited number of time-steps
            
            start = batch_idx * batch_size
            stop = start + batch_size

            batchX = x_train[start:stop]
            batchY = y_train[start:stop]

            #batchX = sess.run(tf.nn.l2_normalize(batchX, 0, epsilon=1e-12, name=None))


            #a = np.random.randint(2, size= (10,6))
            #run the computation graph, give it the values
            #we calculated earlier
            _total_loss, _train_step, _current_state, _predictions_series,_accuracy = sess.run(
                [total_loss, train_step, current_state, predictions_series, accuracy],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state
                })

            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                #print("prediction:",_predictions_series )
                #plot(loss_list, _predictions_series, batchX, batchY)
        print "Epoch Accuracy:", sess.run([accuracy], feed_dict={batchX_placeholder:x_test[0:batch_size], batchY_placeholder:y_test[0:batch_size],init_state:_current_state})
#plt.ioff()
#plt.show()