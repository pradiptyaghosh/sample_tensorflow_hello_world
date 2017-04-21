import tensorflow as tf

'''
input > weight > hidden layer 1(activation function) > weights > hidden l2
(activation function) > weights > output layer

compare output to intended output > cost or loss function ( cross entropy)
optimization function(optimizer) > minimize cost (AdamOptimizer...SGD, AdaGrad)

backpropogation

feed forward + backprop = epoch


'''

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
'''
0 = [1,0,0,0,0,0,0,0,0]
'''

hm_epochs = 20
n_class = 10
batch_size = 128

chunk_size = 28
n_chunks = 28

rnn_size = 512

# height * weight
x = tf.placeholder('float',[None,n_chunks,chunk_size])
y = tf.placeholder('float')


def recurrent_network_model(x):
	layer = { 'weights':tf.Variable(tf.random_normal([rnn_size,n_class])),
				'biases':tf.Variable(tf.random_normal([n_class])) }

	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x,[-1,chunk_size])
	x = tf.split(0, n_chunks, x)

	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
	outputs, states = rnn.rnn(lstm_cell,x,dtype=tf.float32)

	output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

	return output

def train_neural_network(x):
	prediction = recurrent_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

	optimizer = tf.train.AdamOptimizer().minimize(cost)
#	optimizer = tf.train.MomentumOptimizer(0.00001, momentum = 0.5).minimize(cost)
#	optimizer = tf.trainSGDOptimizer().minimize(cost)
#	optimizer = tf.train.AdagradOptimizer(0.5).minimize(cost)


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
	
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

				_,c = sess.run([optimizer,cost], feed_dict = {x:epoch_x, y:epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of',hm_epochs,'loss', epoch_loss)
			
		correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))	
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape(-1, n_chunks,chunk_size), y:mnist.test.labels}))


train_neural_network(x)
