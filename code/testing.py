import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from clamity_env import ClamEnv


# Set learning parameters
y = 1
e0 = 0.3
e = e0
num_episodes = 1500
runs = 10
lr = 0.1

env = ClamEnv()
num_states = env.width*env.height
num_actions = len(env.action_space.actions)

tf.reset_default_graph()


#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,num_states],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([num_states,num_actions],0,0.01))
Qout = tf.matmul(inputs1,W)
#h = tf.layers.dense(inputs1, 16, tf.nn.relu)
#Qout = tf.layers.dense(h, num_actions)

predict = tf.argmax(Qout,1)


#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,num_actions],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=lr)
updateModel = trainer.minimize(loss)


init = tf.global_variables_initializer()

#create lists to contain total rewards and steps per episode
returns = np.zeros(num_episodes)
avg_returns = np.zeros(num_episodes)
with tf.Session() as sess:
    sess.run(init)
    for run in range(runs):
        print(run)
        for i in range(num_episodes):
            #Reset environment and get first new observation
            s = env.reset()
            G = 0
            d = False
            t = 0

            while t < 100:   # each episode is at most 100 time steps long
                t+=1
                #Choose an action by greedily (with e chance of random action) from the Q-network
                a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(num_states)[s:s+1]})
                if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()
                #Get new state and reward from environment
                s1,r,d,_ = env.step(a[0])
                #Obtain the Q' values by feeding the new state through our network
                Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(num_states)[s1:s1+1]})
                #Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,a[0]] = r + y*maxQ1
                #Train our network using target and predicted Q values
                sess.run([updateModel],feed_dict={inputs1:np.identity(num_states)[s:s+1],nextQ:targetQ})
                G += r + y*G
                s = s1
                if d == True:
                    e = e0-e0*i/num_episodes
                    break
            returns[i] = G
        avg_returns += (returns - avg_returns)/(run+1)

plt.plot(avg_returns)
plt.show()
