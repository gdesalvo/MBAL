import random
import math
import sys
import matplotlib
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 22})
np.random.seed(34223)


def experts_rewards(K,T,process_type,num_folds):
    #pre-calculate all rewards for different settings
    #returns reward matrix where row is time and column is experts. That is each row contains the all the experts rewards at time t.
    # time starts from 1!
    rew={}

    if process_type==1:
        #IID.
        mean=np.linspace(0.01,0.99,K) #weird to have prob=1 or prob=0 for binomial
        np.random.shuffle(mean)
        for fold in range(num_folds):
            rewards=[]
            for t in range(1,T+1):
                rewards.append([  np.random.binomial(1, i, 1)[0] for i in mean])
            rew[fold]=np.array(rewards)
        return rew

    elif process_type==2:
        #rotting arms
        theta=np.linspace(0.5,10,num=K) 
        np.random.shuffle(theta)
#        muc=np.linspace(0,0.5,num=K) 
#        np.random.shuffle(muc)
        muc=[0]*K  # should prob just remove this
        for fold in range(num_folds):
            rewards=[]
            for curr_times in range(1,T+1):
                rewards.append([ np.random.binomial(1,muc[j]+math.pow(curr_times,-theta[j]),1)[0]  for j in range(len(theta))]) 
            rew[fold]=np.array(rewards)
        return rew

    elif process_type==3:

        #rarely changing means 
        num_mean_changes=np.random.randint(1,10,K)
        
        mean_time_change=[]
        means=[]

        for arm in range(K):
            mean_time_change.append(np.linspace(1,T+1,num_mean_changes[arm])) 
            means.append( [random.uniform(0.01,0.99) for r in xrange(num_mean_changes[arm])] ) 
        for fold in range(num_folds):
            rewards=[]
            for curr_times in range(1,T+1):
                rewards.append([ np.random.binomial(1,means[j][np.argmax(mean_time_change[j]>curr_times)],1)[0]  for j in range(K)])
            rew[fold]=np.array(rewards)

        return rew,mean_time_change

    elif process_type==4:
        #drifting
        prev=[random.uniform(0.01+1.0/math.sqrt(T),0.99-1.0/math.sqrt(T)) for r in range(K)]
        pm=[np.random.choice([-1.0, 1.0], size=(K,), p=[1./2, 1./2]) for tt in range(T+1)]

#        offset=[random.uniform(0.0001,1) for r in xrange(K)]
#        rewards.append([prev[i]+offset[i] for i in range(K)])
        for fold in range(num_folds):
            rewards=[]
            for curr_times in range(1,T+1):
                rewards.append([ np.random.binomial(1,prev[j]+math.pow(T,-3.0/2.0)*pm[curr_times][j],1)[0] for j in range(K)])
                prev=[prev[j]+math.pow(T,-3.0/2.0)*pm[curr_times][j] for j in range(K)]
            rew[fold]=np.array(rewards)
        return rew

    elif process_type==6:
        #random processes FIX BELOW. 
        # Y_t=Y_{t-1}*alpha_t+ guassian noise
        alpha_change= np.linspace(1.0, T+1, num=K)
        rewards.append([0.0]*K)
        for curr_times in range(1,T+1):        
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
        return np.array(rewards)
    else:
        print 'not avaialbe type'
        return []
        
    



        


def calc_weights_q(curr_time,process_type,mtt_change):
    if process_type ==1 or process_type==2 or process_type==4:
        #IID, rotting arms, drifting
        q=(1.0/curr_time)*np.ones(int(curr_time)) #FIX THIS

    elif process_type==3: #add input to function mean_time_change and arm
#        #changing means DOUBLE CHECK THIS

        q=(1.0/(curr_time-mtt_change[np.argmax(mtt_change>curr_time)-1] +1))*np.ones(int(curr_time))

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

def slack(num_pulled,beta,time):
    return math.sqrt(beta*math.log(time)/(2.0*num_pulled) )


def exp3(K,T,process_type,exp_rewards):
    eta=1.0/math.sqrt(float(K*T))
    prob=[1.0/float(K)]*K
    L=[0]*K
    expert_pulls=[0]*K
    reward_alg = [0]
    for t in range(T):
        arm=np.random.choice(np.arange(0,K), p=prob)
        curr_rew=exp_rewards[int(expert_pulls[arm])][arm]
        reward_alg.append((reward_alg[-1]*float(t)+curr_rew)/float(t+1))
        L[arm]+=curr_rew/prob[arm]
        expert_pulls[arm]+=1
        prob[arm]=math.exp(eta*L[arm])
        tmpp=prob
        prob=[jj/sum(tmpp) for jj in tmpp]

    return reward_alg


def ucb(K,T,process_type,exp_rewards): 
    beta=3.0 #beta of UCB confidence. need to update this alg. deprecated

    # #initialization step
    reward_alg = [0]
    emp_avg=[0]*K
    for i in range(K):
        emp_avg[i]=exp_rewards[0][i]
        reward_alg.append( (reward_alg[-1]*float(i)+exp_rewards[0][i])/float(i+1))
        expert_pulls = [1.0] * K #everyone is pulled once in intiliazation step

    for t in range(K, T):
        #find best arm
        ucb_list = [emp_avg[i] + slack(expert_pulls[i],beta,t) for i in range(K)] #soinefficient
        best_arm = ucb_list.index(max(ucb_list)) 

        best_expert_reward=exp_rewards[int(expert_pulls[best_arm])][best_arm]

        expert_pulls[best_arm] += 1 #update number of times arm is pulled
        #update emp average of expert
        inv_pull = 1.0 / expert_pulls[best_arm]
        emp_avg[best_arm] = best_expert_reward * inv_pull + (1-inv_pull) * emp_avg[best_arm]

        #update regret
        reward_alg.append((reward_alg[-1]*float(t)+best_expert_reward)/float(t+1))

    return reward_alg


def disc_ucb(K,T,process_type,exp_rewards,mt_changes): 
    beta=3.0 #beta of UCB confidence

#    exp_rewards=experts_rewards(K,T,4)
#    print exp_rewards
    #initialization step

    reward_alg = [0]
    emp_avg=[0]*K
    for i in range(K):
        emp_avg[i]=[exp_rewards[0][i]]
        reward_alg.append( (reward_alg[-1]*float(i)+exp_rewards[0][i])/float(i+1))
        expert_pulls = [1.0] * K #everyone is pulled once in intiliazation step

    for t in range(K, T):
        #find best arm
        weights_q= [calc_weights_q(expert_pulls[i],process_type,mt_changes[i])    for i in range(K) ]
        ucb_list = [calc_emp_avg(emp_avg[i],weights_q[i]) + calc_slack(weights_q[i],beta,t) for i in range(K)] #soinefficient
        best_arm = ucb_list.index(max(ucb_list))

        best_expert_reward=exp_rewards[int(expert_pulls[best_arm])][best_arm]
        expert_pulls[best_arm] += 1 #update number of times arm is pulled

        #update emp average of expert
        emp_avg[best_arm].append(best_expert_reward)


        #update regret
        reward_alg.append((reward_alg[-1]*float(t)+best_expert_reward)/float(t+1))

    return reward_alg 





if __name__ == "__main__":

    K=100 #experts
    T=500 #time horizon
    NUM_FOLDS=50
    taxis=range(K,T+1)

    types = {1: 'i.i.d', 2: 'Rotten Arms', 3: 'Rarely Changing Means', 4: 'Drifting'}
    for process_type in range(1,5):
        print process_type
        if process_type!=3:
            exp_rewards=experts_rewards(K,T,process_type,NUM_FOLDS)

            mean_tchanges=[0]*K
            de=[]
            ee=[]
            for fd in range(NUM_FOLDS):
                de.append(disc_ucb(K,T,process_type,exp_rewards[fd],mean_tchanges)[K:])
                ee.append(exp3(K,T,process_type,exp_rewards[fd])[K:])
            de=np.array(de)
            ee=np.array(ee)

            plt.plot(taxis, np.mean(de, axis=0), 'r-', taxis,np.mean(ee, axis=0), 'b-',linewidth=4.0)
            plt.plot(taxis, np.mean(de, axis=0)+np.std(de, axis=0), 'r--', taxis,np.mean(ee, axis=0)+np.std(ee, axis=0), 'b--')
            plt.plot(taxis, np.mean(de, axis=0)-np.std(de, axis=0), 'r--', taxis,np.mean(ee, axis=0)-np.std(ee, axis=0), 'b--')
            plt.xlabel('Rounds')
            plt.ylabel('Average Rewards')
            plt.title(types[process_type])
            plt.savefig('./process_type_'+types[process_type]+'.png', dpi=1000,bbox_inches='tight')


        else:
            exp_rewards,mean_tchanges=experts_rewards(K,T,process_type,NUM_FOLDS)
            de=[]
            ee=[]
            for fd in range(NUM_FOLDS):
            
                de.append(disc_ucb(K,T,process_type,exp_rewards[fd],mean_tchanges)[K:])
                ee.append(exp3(K,T,process_type,exp_rewards[fd])[K:])
            de=np.array(de)
            ee=np.array(ee)

            plt.plot(taxis, np.mean(de, axis=0), 'r-', taxis,np.mean(ee, axis=0), 'b-',linewidth=4.0)
            plt.plot(taxis, np.mean(de, axis=0)+np.std(de, axis=0), 'r--', taxis,np.mean(ee, axis=0)+np.std(ee, axis=0), 'b--')
            plt.plot(taxis, np.mean(de, axis=0)-np.std(de, axis=0), 'r--', taxis,np.mean(ee, axis=0)-np.std(ee, axis=0), 'b--')
            plt.xlabel('Rounds')
            plt.ylabel('Average Rewards')
            plt.title(types[process_type])
            plt.savefig('./process_type_'+types[process_type]+'.png', dpi=1000,bbox_inches='tight')




    



    
   # print emp_avg(allrewards[:curr_time,1],wq)
   # print slack(wq,curr_time,4)
    
