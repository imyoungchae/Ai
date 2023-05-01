'''
state = 미로의 각 위치 (네모)
Action = 행동 (상하좌우)
Reward = Goal에 도착하면 + 5 / 한번 이동시마다 -1
Policy = 어떤 행동에서 각 행동을 할 확률들 
Trainsition = 어떤 행동을 했을 때 다음 특정 state로 갈 확률 



'''
import time
import numpy as np
import matplotlib.pyplot as plt 

class Env:
    def __init__(self):
        self.grid_width=10
        self.grid_height=self.grid_width
        self.action_grid=[(-1,0),(1,0),(0,-1),(0,1)]         
        self.goal=[9,9]
        
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
        return[0,0]

                 
class MC_agent:
    def __init__(self):
        self.action_grid=[(-1,0),(1,0),(0,-1),(0,1)] # Up/Down/Left/Right action_grid type: list
        self.action_text=['U','D','L','R']
        self.grid_width=10
        self.grid_height=self.grid_width
        self.value_table =np.zeros((self.grid_width,self.grid_height)) # value_table type: np
            
        self.e=.1
        '''학습 초반에 ε의 값을 1로 설정한다. 따라서 에이전트는 100% 무작위 탐색을 수행한다. 
        그리고 학습이 진행됨에 따라 조금씩 ε 값을 줄여나간다. 
        이에 따라 무작위 탐색을 수행할 확률은 조금씩 줄어들고 탐욕적 행동을 할 확률이 조금씩 늘어난다.
        탐욕적 방법은 주어진 시점에 에이전트가 가장 큰 보상을 줄 것이라고 기대하는 행동만을 선택하는 것'''
            
        self.learning_rate=.00001 
            
        self.discount_factor=.95 
        '''discount_factor: 미래의 보상을 얼만큼 고려할 것인가. 
        0과 1 사이의 값만 가질 수 있으며 보상에 곱해지면 보상이 줄어든다. 
        통상적으로 $\gamma$라고 표기. 1에 가까울수록 미래의 보상에 많은 가중치를 두는 것'''
            
        self.memory=[]
        
    def get_action(self,state):
        if np.random.randn()<self.e:
            idx=np.random.choice(len(self.action_grid),1)[0]
        else:
            next_values=np.array([])
            for s in self.next_states(state):
                next_values=np.append(next_values,self.value_table[tuple(s)])
            max_value=np.amax(next_values)
            tie_Qchecker=np.where(next_values==max_value)[0]
            
            if len(tie_Qchecker)>1:
                idx=np.random.choice(tie_Qchecker,1)[0]
            else:
                idx=np.argmax(next_values)
        action=self.action_grid[idx]
        return action
      
    def next_states(self,state):
        x,y=state
        next_S=[]
        for action in self.action_grid:
            x+=action[0]
            if x<0:
                x=0
            elif x>9:
                x=9
            y+=action[1]
            if y<0:
                y=0
            elif y>9:
                y=9
            next_S.append([x,y])
        return next_S
    
    # using First visit MC  
    '''
    
    - first visit MC: 에피소드에서 하나의 state를 여러번 지나갈 수 있음
    이때 해당 state에 첫번째 방문 했을때의 value만을 사용하는 방식
    에피소드가 여러번 진행이 되므로 각 에피소드에 대한 평균으로 value를 추정
    에피소드가 충분히 무한대로 진행이 되게 되면 
    이렇게 평균값으로 추정한 value가 최적화된 실제 value와 같게 되고 
    이를 통해서 policy를 업데이트 하면 우리가 원하는 최적의 정책을 찾을 수 있게됨
    
    - every visit MC: 하나의 states를 두번이상 지나갔다면 
    이때의 모든 value를 각각 사용하여 평균내어 추정하는 방식
    
    '''
    def update(self):
        G_t=0                       
        visit_states=[]
        for sample in reversed(self.memory):
            state=sample[0]
            reward=sample[1]
            if state not in visit_states:
                visit_states.append(state)
                G_t=reward+self.discount_factor*G_t
                V_t=self.value_table[tuple(state)]
                
                # update Value
                self.value_table[tuple(state)]=V_t+self.learning_rate*(G_t-V_t)
                
    
    def memorizer(self,state,reward,done):
        self.memory.append([state,reward,done])
    
    def print_reward(self):
        for experience in self.memory:
            print('reward: ',experience[1])
            
    def save_actionseq(self,action_sequence,action):
        idx=self.action_grid.index(action)
        action_sequence.append(self.action_text[idx])

if __name__=="__main__":
    env=Env()
    agent=MC_agent()
    total_episode=50000
    sr=0
    episode_list=[]
    reward_list=[]
    num_step=[]
    start = time.time()

    for episode in range(total_episode):
        action_sequence=[]
        total_reward=0 
        state=env.reset()  
        action=agent.get_action(state)
        done=False
        walk=0
        
        while True:
            next_state,reward,done=env.step(state,action)
            agent.memorizer(state,reward,done)
            agent.save_actionseq(action_sequence,action)
            walk+=1
            
            state=next_state
            action=agent.get_action(state)
            next_action=agent.get_action(next_state)
            total_reward=total_reward+reward
            
            if done:
                if episode%100==0:
                    agent.print_reward()
                    print('finished at ', state)
                    print('episode: {}\nThe number of step: {}\nThe sequence of action is {} \nThe total reward is: {}'.format(episode,walk,action_sequence,total_reward))
                    print('\n')
                    print('----------------------------------------------\n')
                    reward_list.append(total_reward)
                    episode_list.append(episode)
                    num_step.append(walk)
                        
                if state==env.goal:
                    sr+=1
                        
                agent.update()
                agent.memory.clear()
                break
            
print('total episode: ',total_episode) 
print('sr : ',sr)     
print('The accuary: ',sr/total_episode*100,'%')
code_time=time.time()-start


plt.plot(episode_list,num_step,'-r',label='num_step')
plt.xlabel('episode')
plt.ylabel('number of step')
plt.title('Monte Carlo Number of step')
plt.text(1,1,'running time: {} sec\ntotal episode: {}'.format(round(code_time,2),total_episode))
plt.grid() 
plt.legend(loc='upper right')
plt.show()
    



