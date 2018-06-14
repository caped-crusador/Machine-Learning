## Reinforcement learning in 1d maze
import numpy as np
import matplotlib.pyplot as plt


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

gamma=0.5;

print('--> Analytic solution for optimal policy')

# Defining reward vector R
i = 0
R = np.zeros(10)
for s in range(0, 5):
    for a in range(-1, 2, 2):
        R[i] = rho(s, a)
        i += 1

# Defining transition matrix
T=np.zeros([10,10]);
T[0,0]=1; T[1,1]=1; T[2,0]=1; T[3,5]=1; T[4,2]=1
T[5,7]=1; T[6,5]=1; T[7,9]=1; T[8,8]=1; T[9,9]=1

# Calculate Q-function
Q=np.linalg.inv(np.eye(10)-gamma*T) @ np.transpose(R)
Q=np.reshape(Q,[5,2])

policy=calc_policy(Q)
print('Q values: \n',np.transpose(Q))
print('policy: \n',np.transpose(policy))
Qana=Q
print(R)


# Dynamic programming
print('\n--> Dynamic Programing')


Q=np.zeros([5,2])
for iter in range(20):
    for s in range(0,5):
        for a in range(-1,2,2):
            act = np.int(policy[tau(s,a)])
            Q[s,idx(a)]=rho(s,a)+gamma*Q[tau(s,a),idx(act)]

policy=calc_policy(Q)
print('Q values: \n',np.transpose(Q))
print('policy: \n',np.transpose(policy))



print('\n--> Policy iteration')

Q = np.zeros([5, 2])
policy = calc_policy(Q)
for iter in range(3):
    for s in range(0, 5):
        for a in range(-1, 2, 2):
            act = np.int(policy[tau(s, a)])
            Q[s, idx(a)] = rho(s, a) + gamma * Q[tau(s, a), idx(act)]
    policy = calc_policy(Q)

print('Q values: \n', np.transpose(Q))
print('policy: \n', np.transpose(policy))


print('\n--> Q-iteration')

Q_new=np.zeros([5,2])
Q=np.zeros([5,2])
policy = np.zeros(5)
for iter in range(2):
    for s in range(0,5):
        for a in range(-1,2,2):
            maxValue = np.maximum(Q[tau(s,a),0],Q[tau(s,a),1])
            Q_new[s,idx(a)]=rho(s,a)+gamma*maxValue
    Q=np.copy(Q_new)

policy=calc_policy(Q)
print('Q values: \n',np.transpose(Q))
print('policy: \n',np.transpose(policy))