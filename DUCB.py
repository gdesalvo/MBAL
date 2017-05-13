import random
import math
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 22})
random.seed(3428)




def experts_rewards(K,T,process_type):
    #pre-calculate all rewards for different settings
    #returns reward matrix where row is time and column is experts. That is each row contains the all the experts rewards at time t.
    # time starts from 1!
    rewards=[]
    if process_type==1:
        #IID.
        mean=np.linspace(5,10,K) 
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



        


def weights_q(curr_time,process_type):
    if process_type ==1:
        #IID
        q=(1.0/curr_time)*np.ones(curr_time)

    elif process_type==2:
        #rotting arms
        q=(1.0/curr_time)*np.ones(curr_time)

    elif process_type==6:
     #general process
         q=(1.0/curr_time)*np.ones(curr_time) #FIX THIS
    else:
        print 'not available'

    return q


def emp_avg(rewards,weights_q):
     return sum(rewards[:]*weights_q[:])
     

def slack(weight_q,time,beta):
    return np.linalg.norm(weight_q)*math.sqrt(beta*math.log(time))


def ucb(K,T,process_type): 
    
    exp_rewards=experts_rewards(K,T,process_type)
   #TODO WRITE THE UCB ALGORITHM BELOW. 

    # #initialization step
    # reward_alg = 0

    # for i in range(K):
    #     exp_reward = rej_loss(dat[i][1], expert_label, c)
    #     expert_avg.append(exp_reward)
    #     loss_alg += exp_loss
    #     expert_pulls = [1.0] * K #everyone is pulled once in intiliazation step

    # for t in range(K, T):
    #     #find best arm
    #     ucb_list = [emp_avg[i] + slack[i] for i in range(K)] #soinefficient
    #     best_arm = ucb_list.index(max(ucb_list)) 


    #     expert_pulls[best_arm] += 1 #update number of times arm is pulled

    #     #update loss of best arm average
    #     inv_pull = 1.0 / expert_pulls[best_arm]
    #     expert_loss = rej_loss(dat[t][1], exp_label(dat[t][0], experts[best_arm]), c)
    #     expert_avg[best_arm] = expert_loss * inv_pull + (1-inv_pull) * expert_avg[best_arm]
    #     if exp_label(dat[t][0],experts[best_arm]) == -1:
    #         count_rej+=1

    #     #update regret
    #     loss_alg += expert_loss

    # return loss_alg / float(T) , count_rej/float(T)




if __name__ == "__main__":

    K=5 #experts
    T=50 #time horizon
    curr_time=4
    process_type=3
    prev_experts=range(K)

    allrewards=experts_rewards(K,T,process_type)
#    wq=weights_q(curr_time,process_type)
#    print allrewards


    
   # print emp_avg(allrewards[:curr_time,1],wq)
   # print slack(wq,curr_time,4)
    
