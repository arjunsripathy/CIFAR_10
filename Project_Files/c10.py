import numpy as np
import tensorflow as tf


USE_OLD = False

QUICKRUN= False
useD = 20000
SHIFT = 0
if(QUICKRUN):
	useD = 100

costBatch = 100

useT = 10000
if(QUICKRUN):
	useT = 100

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

a = unpickle("cifar-10-batches-py/data_batch_1")
b = unpickle("cifar-10-batches-py/data_batch_2")
c = unpickle("cifar-10-batches-py/data_batch_3")
d = unpickle("cifar-10-batches-py/data_batch_4")
e = unpickle("cifar-10-batches-py/data_batch_5")
t = unpickle("cifar-10-batches-py/test_batch")

#1000,3072
rawImageData = np.concatenate((a["data"],b["data"],c["data"],d["data"],e["data"]))[SHIFT:SHIFT+useD]

#1000,1024
red = rawImageData[:,:1024]
green = rawImageData[:,1024:2048]
blue = rawImageData[:,2048:]

#1000*1024
redf = np.reshape(red,[useD*1024])
bluef = np.reshape(blue,[useD*1024])
greenf = np.reshape(green,[useD*1024])

#1000*1024,3
zf = zip(redf,bluef,greenf)

#1000,1024,3
imageData = np.reshape(zf,[useD,1024,3])

rawTestImageData = t["data"][:useT]

red = rawTestImageData[:,:1024]
green = rawTestImageData[:,1024:2048]
blue = rawTestImageData[:,2048:]

redf = np.reshape(red,[useT*1024])
bluef = np.reshape(blue,[useT*1024])
greenf = np.reshape(green,[useT*1024])

zf = zip(redf,bluef,greenf)

testImageData = np.reshape(zf,[useT,1024,3])


rawImageLabels = np.concatenate((a["labels"],b["labels"],c["labels"],d["labels"],e["labels"]))[SHIFT:SHIFT+useD]

rawTestImageLabels = t["labels"][:useT]

wStdDev = 0.01
bInit= 0.1
numH = 1000
EPOCHS = 6
MBSIZE  = 10
LEARNING_RATE = 3e-4
DROPOUT = 0.5

nD = len(imageData)
nT = len(testImageData)

def weight(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev=wStdDev))

def bias(shape):
	return tf.Variable(tf.constant(bInit,shape=shape))

def convL(x,W,b):
	return tf.nn.relu(tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')+b)

def mp(l):
	return tf.nn.max_pool(l,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

def encode(v):
	result = []
	for i in range(len(v)):
		a = np.zeros([10])
		a[v[i]]=1
		result.append(a)
	return(result)

imageLabels = encode(rawImageLabels)

testImageLabels = encode(rawTestImageLabels)


x = tf.placeholder(tf.float32,[None,1024,3])

xR = tf.reshape(x,[-1,32,32,3])
#32x32

y_ = tf.placeholder(tf.float32,[None,10])


Wc1 = weight([5,5,3,32])
bc1 = bias([32])


hc1 = convL(xR,Wc1,bc1)
#28x28
hp1 = mp(hc1)
#14x14

Wc2 = weight([5,5,32,64])
bc2 = bias([64])


hc2 = convL(hp1,Wc2,bc2)
#10x10
hp2 = mp(hc2)
#5x5

Wc3 = weight([5,5,64,128])
bc3 = bias([128])

hc3 = convL(hp2,Wc3,bc3)

hcFlat = tf.reshape(hc3,[-1,128])

Wf1 = weight([128,numH])
bf1 = bias([numH])

hf1 = tf.nn.relu(tf.matmul(hcFlat,Wf1)+bf1)

keep_prob = tf.placeholder(tf.float32)
hf1d = tf.nn.dropout(hf1,keep_prob)

Wf2 = weight([numH,10])
bf2 = bias([10])

y = tf.matmul(hf1d,Wf2)+bf2
ySoft = tf.nn.softmax(y)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
meanRight = 10*tf.reduce_mean(tf.multiply(ySoft,y_))

saver = tf.train.Saver()


with tf.Session() as sess:
	if(USE_OLD):
		saver.restore(sess,"/tmp/tfc10.ckpt")
	else:
		sess.run(tf.global_variables_initializer())

	for i in range(EPOCHS):

		for j in range(nD/MBSIZE):
			iDB = imageData[j*(MBSIZE):(j+1)*MBSIZE]
			iL = imageLabels[j*(MBSIZE):(j+1)*MBSIZE]
			train_step.run(feed_dict={x:iDB,y_:iL,keep_prob:DROPOUT})


		tcE = 0
		tacc = 0
		tmR = 0
		for j in range(nD/costBatch):
			iDB = imageData[j*(costBatch):(j+1)*costBatch]
			iL = imageLabels[j*(costBatch):(j+1)*costBatch]
			[cE,acc,mR] = sess.run([cross_entropy,accuracy,meanRight],feed_dict={x:iDB,y_:iL,keep_prob:1.0})
			tcE += cE
			tacc += acc
			tmR += mR

		tcE /= (nD/costBatch)
		tacc /= (nD/costBatch)
		tmR /= (nD/costBatch)
			
		print("TRAINING DATA")
		print("EPOCH %i: COST: %f, ACCURACY: %f, MEAN RIGHT: %f"%(i+1,tcE,tacc,tmR))


		tcE = 0
		tacc = 0
		tmR = 0
		for j in range(nT/costBatch):
			iDB = testImageData[j*(costBatch):(j+1)*costBatch]
			iL = testImageLabels[j*(costBatch):(j+1)*costBatch]
			[cE,acc,mR] = sess.run([cross_entropy,accuracy,meanRight],feed_dict={x:iDB,y_:iL,keep_prob:1.0})
			tcE += cE
			tacc += acc
			tmR += mR

		tcE /= (nT/costBatch)
		tacc /= (nT/costBatch)
		tmR /= (nT/costBatch)
			
		print("TEST DATA")
		print("EPOCH %i: COST: %f, ACCURACY: %f, MEAN RIGHT: %f"%(i+1,tcE,tacc,tmR))

		saver.save(sess, "/tmp/tfc10.ckpt")
		print("SAVED")

