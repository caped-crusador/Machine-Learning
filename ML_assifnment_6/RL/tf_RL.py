import numpy as np
import random
import tensorflow as tf

# inputs
input_1 = tf.placeholder(shape=[1, 5], dtype=tf.float32)
weights = tf.Variable(tf.random_uniform([5, 2], 0, 0.01))
Q_out = tf.matmul(input_1, weights)
predict_policy = tf.argmax(Q_out, 1)

# loss calculation
Q_new = tf.placeholder(shape=[1,2], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Q_new - Q_out))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
modelUpdate = trainer.minimize(loss)

init = tf.initialize_all_variables()
# parameters
gamma = 0.5
iterations = 10


def tau(s, a):
    if s == 0 or s == 4:
        return s
    else:
        return s+a


def rho(s, a):
    return (s == 1 and a == -1)+2*(s == 3 and a == 1)


def calc_policy(Q):
    policy = np.zeros(5)
    for s in range(0, 5):
        action_idx = np.argmax(Q[s, :])
        policy[s] = 2*action_idx-1
        policy[0] = policy[4] = 0
    return policy.astype(int)


def idx(a):
    return int((a+1)/2)


with tf.Session() as sess:
    sess.run(init)

    for iter in range(iterations):
        s = 2
        rList = []
        rAll = 0
        d = False
        j = 0

        while j < 99:
            j += 1

            # choose an action
            a, allQ = sess.run([predict_policy, Q_out], feed_dict={input_1:np.identity(5)[s:s+1]})
            # sess.run(print(a))
            # sess.run(print(allQ))
            s1 = tau(s, a)
            # get new state and reward from env


            r = rho(s, a)

            # obtain Q values bu feeding new state through network
            Q1 = sess.run(Q_out, feed_dict={input_1:np.identity(5)[s:s+1]})

            Q1_max = np.max(Q1)
            targetQ = allQ
            targetQ[0, idx(a)] = r + gamma*Q1_max
            # sess.run(print(targetQ))

            # train network
            _,W1 = sess.run([modelUpdate, weights], feed_dict={input_1:np.identity(5)[s:s+1],
                                                         Q_new:targetQ})
            rAll += r
            s = s1

        rList.append(rAll)





        # for s in range(0, 5):
        #     for a in range(-1, 2, 2):
        #         maxValue = np.maximum(Q[tau(s, a), 0], Q[tau(s, a), 1])
        #         Q_new[s, idx(a)] = rho(s, a) + gamma * maxValue
        # Q = np.copy(Q_new)



