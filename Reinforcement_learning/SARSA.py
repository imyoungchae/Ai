import numpy as np
import matplotlib.pyplot as plt 
import time

class Env:
    def __init__(self):
        self.grid_width = 10
        self.grid_height = self.grid_width
        self.action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)] # U, D, L, R
            
        self.obstacles=[[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,9],
                    [1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,9],
                    [2,9],
                    [3,0],[3,1],[3,2],[3,3],[3,5],[3,6],[3,7],[3,8],[3,9],
                    [4,0],[4,1],[4,2],[4,3],[4,5],[4,6],[4,7],[4,8],[4,9],
                    [5,0],[5,1],[5,2],[5,3],[5,5],[5,6],[5,7],[5,8],[5,9],
                    [6,0],[6,1],[6,8],[6,9],
                    [7,3],[7,4],[7,5],[7,6],[7,8],[7,9],
                    [8,0],[8,1],[8,3],[8,4],[8,5],[8,6],
                    [9,0],[9,1]]
        
        self.boom=[[0,8],[7,0]]
        self.goal = [9,9]
        self.semi_goal=[5,5]
        self.semi_goal2=[[8,7],[9,7]]
        
    def step(self,state,action):
        x,y=state
        x+=action[0]
        y+=action[1]
        
        if x<0:
            x=0
        elif x>(self.grid_width-1):
            x=(self.grid_width-1)
            
        if y<0:
            y=0
        elif y>(self.grid_width-1):
            y=(self.grid_width-1)

        next_state=[x,y]
        
        if next_state in self.obstacles:
            reward = -1
            done =False 
            
        elif next_state in self.boom:
            reward=-5
            done=True   
                              
        elif next_state==self.goal:
            reward = 50
            done=True
            
        else:
            reward=0
            done=False
        
        return next_state,reward,done
    
    def reset(self):
        return [0, 0]
    
class SARSA_agent:
    def __init__(self):
        self.action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_text= ['U', 'D', 'L', 'R']
        self.grid_width = 10
        self.grid_height = self.grid_width
        self.Qtable = np.zeros((self.grid_width, self.grid_height, len(self.action_grid)))
        self.e = .01
        self.learning_rate = .001
        self.discount_factor = .9
        self.memory=[]

    def get_action(self, state):
        # with prob.ε take random action
        if np.random.randn() <  self.e :
            idx = np.random.choice(len(self.action_grid),1)[0]
        else :
            Qvalues = self.Qtable[tuple(state)]
            maxQ = np.amax(Qvalues)
            tie_Qchecker = np.where(Qvalues==maxQ)[0]
            
            # if tie max value, get random
            if len(tie_Qchecker) > 1:
                idx = np.random.choice(tie_Qchecker, 1)[0]
            else :
                idx = np.argmax(Qvalues)
                
        action = self.action_grid[idx]
        return action    
        
    # using First visit MC    
    def update(self, state, action, reward, next_state, next_action):
        action_idx = self.action_grid.index(action)
        next_action_idx = self.action_grid.index(next_action)
        current_Q = self.Qtable[tuple(state)][action_idx]
        next_Q = self.Qtable[tuple(next_state)][next_action_idx]
        updated_Q = current_Q + self.learning_rate*((reward + self.discount_factor*next_Q)-current_Q)
        self.Qtable[tuple(state)][action_idx] = updated_Q
        
    def save_actionseq(self, action_sequence, action):
        idx = self.action_grid.index(action)
        action_sequence.append(self.action_text[idx])



        
        
if __name__ =='__main__':
    env = Env()
    agent = SARSA_agent()
    total_episode = 50000
    sarsa_sr = 0
    episode_list=[]
    reward_list=[]
    num_step=[]
    start = time.time()
    
    
    for episode in range(total_episode):
        action_sequence=[]
        total_reward = 0
        walk = 0
        
        # initial state, action, done
        state = env.reset()
        action = agent.get_action(state)
        done = False
        
        while not done:
            walk += 1  
            agent.save_actionseq(action_sequence, action)
            
            # next state, action
            next_state, reward, done = env.step(state, action)
            next_action = agent.get_action(next_state)

            # update Qtable
            agent.update(state, action, reward, next_state, next_action)
            
            total_reward += reward
            
            if done:
                if episode % 100 == 0:
                    print('finished at ', state)
                    print('episode: {}\nThe number of step: {}\nThe sequence of action is {} \nThe total reward is: {}'.format(episode,walk,action_sequence,total_reward))
                    print('\n')
                    print('----------------------------------------------\n')
                    reward_list.append(total_reward)
                    episode_list.append(episode)
                    num_step.append(walk)

                    
                if state == env.goal:
                    sarsa_sr += 1

            state = next_state
            action = agent.get_action(state)
            

        # update sarsa_sr if goal is reached
        if state == env.goal:
            sarsa_sr += 1
    code_time=time.time()-start
    print('total episode: ',total_episode) 
    print('sarsa_sr : ',sarsa_sr)           
    print('The accuracy :', sarsa_sr/total_episode*100, '%')       
    print("time :", code_time)


    plt.plot(episode_list,num_step,'-c',label='num_step')
    plt.xlabel('episode')
    plt.ylabel('number of step')
    plt.title('SARSA Number of step')
    plt.text(1,1,'running time: {} sec\ntotal episode: {}'.format(round(code_time,2),total_episode))
    plt.grid() 
    plt.legend(loc='upper right')
    plt.show()
    
