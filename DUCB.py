import random
import math
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 22})
random.seed(34223)




def experts_rewards(K,T,process_type):
    #pre-calculate all rewards for different settings
    #returns reward matrix where row is time and column is experts. That is each row contains the all the experts rewards at time t.
    # time starts from 1!
    rewards=[]
    if process_type==1:
        #IID.
        mean=np.linspace(1,5,K) 
        np.random.shuffle(mean)
        for t in range(1,T):
            rewards.append([np.random.normal(i,0.5) for i in mean])

    elif process_type==2:
        #rotting arms
        theta=np.linspace(0.5,10,num=K) 
        muc=np.linspace(0,5,num=K) 
        for curr_times in range(1,T):
            rewards.append([ muc[j]+math.pow(curr_times,-theta[j])  for j in range(len(theta))])

    elif process_type==3:
        #rarely changing means 
        num_mean_changes=np.random.randint(1,10,K)

        mean_time_change=[]
        means=[]
        for arm in range(K):
            mean_time_change.append(np.linspace(1,T,num_mean_changes[arm])) 
            means.append( [random.uniform(1,5) for r in xrange(num_mean_changes[arm])] ) 

        for curr_times in range(1,T):
            rewards.append([ np.random.normal(means[j][np.argmax(mean_time_change[j]>curr_times)],0.5)  for j in range(K)])
             
    elif process_type==4:
        #drifting
        prev=[random.uniform(1,5) for r in xrange(K)]

#        offset=[random.uniform(0.0001,1) for r in xrange(K)]
#        rewards.append([prev[i]+offset[i] for i in range(K)])
        rewards.append(prev)
        for curr_times in range(1,T):
            pm=np.random.choice([-1.0, 1.0], size=(K,), p=[1./2, 1./2])
            curr=[prev[j]+math.pow(T,-3.0/2.0)*pm[j] for j in range(K)]
            rewards.append(curr)
 #           rewards.append([curr[i]+offset[i] for i in range(K)])
            prev=curr

    elif process_type==6:
        #random processes
        # Y_t=Y_{t-1}*alpha_t+ guassian noise
        alpha_change= np.linspace(1.0, T, num=K)
        rewards.append([0.0]*K)
        for curr_times in range(1,T):        
            noise=np.random.normal(0, 1, K)
            prev_experts=rewards[-1]

            alphat=[]
            for val in alpha_change:
                if val < curr_times:
                    alphat.append(1)
                else:
                    alphat.append(-1)

            rewards.append([prev_experts[j]*alphat[j] + noise[j]  for j in range(K)])
        rewards=rewards[1:]
    else:
        print 'not avaialbe type'


    return np.array(rewards)



        


def calc_weights_q(curr_time,process_type):
    if process_type ==1 or process_type==2 or process_type==4:
        #IID, rotting arms, drifting
        q=(1.0/curr_time)*np.ones(curr_time)

#    elif process_type==3: #add input to function mean_time_change and arm
#        #changing means DOUBLE CHECK THIS
#        q=(1.0/(curr_time-mean_time_change[arm][np.argmax(mean_time_change[arm]>curr_times)] +1))*np.ones(curr_time)

    elif process_type==6:
     #general process
         q=(1.0/curr_time)*np.ones(curr_time) #FIX THIS
    else:
        print 'not available'

    return q


def calc_emp_avg(rewards,weight):
     return sum(rewards[:]*weight[:])
     

def calc_slack(weight_q,beta,time):
    return np.linalg.norm(weight_q)*math.sqrt(beta*math.log(time))

def slack(num_pulled,beta,T):
    return math.sqrt(beta*math.log(T)/(2.0*num_pulled) )


def ucb(K,T,process_type,exp_rewards): 
    beta=3.0 #beta of UCB confidence

    # #initialization step
    reward_alg = 0
    emp_avg=[0]*K
    for i in range(K):
        emp_avg[i]=exp_rewards[0][i]
        reward_alg+= exp_rewards[0][i]
        expert_pulls = [1.0] * K #everyone is pulled once in intiliazation step

    for t in range(K, T):
        #find best arm
        ucb_list = [emp_avg[i] + slack(expert_pulls[i],beta,T) for i in range(K)] #soinefficient
        best_arm = ucb_list.index(max(ucb_list)) 

        best_expert_reward=exp_rewards[expert_pulls[best_arm]][best_arm]
        print best_arm
        expert_pulls[best_arm] += 1 #update number of times arm is pulled
        #update emp average of expert
        inv_pull = 1.0 / expert_pulls[best_arm]
        emp_avg[best_arm] = best_expert_reward * inv_pull + (1-inv_pull) * emp_avg[best_arm]

        #update regret
        reward_alg += best_expert_reward

    return reward_alg / float(T) 


def disc_ucb(K,T,process_type,exp_rewards): 
    beta=3.0 #beta of UCB confidence

#    exp_rewards=experts_rewards(K,T,4)
#    print exp_rewards
    #initialization step
    reward_alg = 0
    emp_avg=[0]*K
    for i in range(K):
        emp_avg[i]=[exp_rewards[0][i]]
        reward_alg+= exp_rewards[0][i]
        expert_pulls = [1.0] * K #everyone is pulled once in intiliazation step

    for t in range(K, T):
        #find best arm
        weights_q= [calc_weights_q(expert_pulls[i],process_type)    for i in range(K) ]
        ucb_list = [calc_emp_avg(emp_avg[i],weights_q[i]) + calc_slack(weights_q[i],beta,t) for i in range(K)] #soinefficient
        best_arm = ucb_list.index(max(ucb_list)) 

        best_expert_reward=exp_rewards[expert_pulls[best_arm]][best_arm]
        expert_pulls[best_arm] += 1 #update number of times arm is pulled

        #update emp average of expert
        emp_avg[best_arm].append(best_expert_reward)
        print best_arm

        #update regret
        reward_alg += best_expert_reward

    return reward_alg / float(T) 





if __name__ == "__main__":

    K=5 #experts
    T=50 #time horizon
  #  curr_time=4
    process_type=1
 #   prev_experts=range(K)

#    allrewards=experts_rewards(K,T,process_type)
#    print allrewards
#    wq=weights_q(curr_time,process_type)
    exp_rewards=experts_rewards(K,T,process_type)
    print exp_rewards
    print ucb(K,T,process_type,exp_rewards)
    print disc_ucb(K,T,process_type,exp_rewards)


    
   # print emp_avg(allrewards[:curr_time,1],wq)
   # print slack(wq,curr_time,4)
    
