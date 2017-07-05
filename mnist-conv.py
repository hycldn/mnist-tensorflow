# -*- coding: utf-8 -*-
# @Author: Rishabh Thukral
# @Date:   2017-07-05 19:14:32
# @Last Modified by:   Rishabh Thukral
# @Last Modified time: 2017-07-05 19:15:14

# coding: utf-8

# # Load Mnist Data

# In[21]:


import numpy as np


# In[22]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# # Strarting TensorFlow interactive session

# In[23]:


import tensorflow as tf
sess = tf.InteractiveSession()


# # Building a Softmax Regression Model
# 
# ## Placholders for input and labels

# In[24]:


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# ## Variables for weights and biases

# In[4]:


W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# In[25]:


sess.run(tf.global_variables_initializer())


# ## Predicted Class and Loss Function
# 

# In[26]:


y = tf.matmul(x,W) + b


# In[27]:


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


# # Training Softmax Regression Model with Gradient Descent

# In[28]:


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[35]:


for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})


# # Evaluating the model

# In[37]:


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# # Building Multilayer Convolution Network
# 
# ## Weight Initialization

# In[39]:


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# ## Convolution and Pooling Operations

# In[41]:


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# ## First Convolution Layer

# In[51]:


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


# In[52]:


x_image = tf.reshape(x, [-1, 28, 28, 1])


# In[53]:


print(x_image)


# In[54]:


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
print(h_conv1)
print(h_pool1)


# ## Second Convolution Layer

# In[56]:


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])


# In[58]:


h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
print(h_conv2)
print(h_pool2)


# ## Densely Connected Layer

# In[60]:


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])


# In[63]:


h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
print(h_pool2_flat)
print(h_fc1)


# ## Dropout

# In[65]:


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# ## ReadOut Layer

# In[67]:


W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])


# In[69]:


y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# ## Training and Evaluating the Model

# In[71]:


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))


# In[73]:


train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# In[74]:


correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[ ]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

