import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#自动下载MNIST数据，并保存在了当前目录下
#下载下来的数据集被分成两部分：60000行的训练数据集（mnist.train）和10000行的测试数据集（mnist.test）

x = tf.placeholder(tf.float32,[None, 784])
y_actual = tf.placeholder(tf.float32,[None, 10])
W = tf.Variable(tf.zeros([784,10]),name='w')       
b = tf.Variable(tf.zeros([10]),name='b')            
y_predict = tf.nn.softmax(tf.matmul(x,W)+b)     
#y_predict为n*10矩阵，每行是该照片的为0~9数字概率分布
#y_actual为n*10 one-hot矩阵

cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))
train= tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)   
#交叉熵（loss）,并用梯度下降法优化

correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))   
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                
#correct_prediction为一维[10]0,1向量，第i元素为1表示第i张图片预测正确

init = tf.global_variables_initializer()
with tf.Session() as sess:
    writer =  tf.summary.FileWriter('./graph', sess.graph)
    sess.run(init)
    for i in range(1000):   
        batch_xs, batch_ys = mnist.train.next_batch(100)           
        #函数DataSet.next_batch()是用于获取以batch_size为大小的一个元组，其中包含了一组图片和标签
        sess.run(train, feed_dict={x: batch_xs, y_actual: batch_ys})   
        if(i%100==0):       
            print("accuracy:",sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))
    writer.close()
            
            
            
            
            
            