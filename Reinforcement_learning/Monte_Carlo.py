'''
state = 미로의 각 위치 (네모)
Action = 행동 (상하좌우)
Reward = Goal에 도착하면 + 5 / 한번 이동시마다 -1
Policy = 어떤 행동에서 각 행동을 할 확률들 
Trainsition = 어떤 행동을 했을 때 다음 특정 state로 갈 확률 

'''

import numpy as np
import matplotlib.pyplot as plt

class Env:
  def __init__(self):
    self.grid_width=10
    self.grid_height=self.grid_width
    self.action_grid=[(-1,0),(1,0),(0,-1),(0,1)] # up,down,right,left
    self.gtraingle1=[1,4]
    self.gtraingle2=[2,4]
    #self.gtraingle3=[3,4]
    #self.gtraingle4=[2,3]
    #self.gtraingle5=[5,1]

    
    #self.gtraingle2=[2,1]
    self.goal=[5,5]
    
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
    
    if next_state==self.gtraingle1 or next_state==self.gtraingle2:
    #if next_state==self.gtraingle1:
        reward=-1
        done =True
    elif next_state==self.goal:
        reward= 3
        done=True
    else:
        reward=0
        done=False
    
    return next_state,reward,done
  
  def reset(self):
      return[0,0]

                 
class MC_agent:
    def __init__(self):
        self.action_grid=[(-1,0),(1,0),(0,-1),(0,1)]
        self.action_text=['U','D','L','R']
        self.grid_width=10
        self.grid_height=self.grid_width
        self.value_table =np.zeros((self.grid_width,self.grid_height)) #10x10
            
        self.e=.1
        # 학습 초반에 ε의 값을 1로 설정한다. 따라서 에이전트는 100% 무작위 탐색을 수행한다. 
        # 그리고 학습이 진행됨에 따라 조금씩 ε 값을 줄여나간다. 
        # 이에 따라 무작위 탐색을 수행할 확률은 조금씩 줄어들고 탐욕적 행동을 할 확률이 조금씩 늘어난다.
        # 탐욕적 방법은 주어진 시점에 에이전트가 가장 큰 보상을 줄 것이라고 기대하는 행동만을 선택하는 것
            
        self.learning_rate=.01 
        #discount_factor이 미래에 대한 신뢰의 문제를 다루었다면, 학습률은 현재 경험에 대한 신뢰의 문제를 다룬다.
            
        self.discount_factor=.95 
        #discount_factor: 미래의 보상을 얼만큼 고려할 것인가. 
        # 0과 1 사이의 값만 가질 수 있으며 보상에 곱해지면 보상이 줄어든다. 
        # 통상적으로 $\gamma$라고 표기. 1에 가까울수록 미래의 보상에 많은 가중치를 두는 것
            
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
            if x<0:
                x=0
            elif x>9:
                x=9
            next_S.append([x,y])
        return next_S
      
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
                self.value_table[tuple(state)]=V_t+self.learning_rate*(G_t-V_t)
    
    def memorizer(self,state,reward,done):
        self.memory.append([state,reward,done])
        
    def save_actionseq(self,action_sequence,action):
        idx=self.action_grid.index(action)
        action_sequence.append(self.action_text[idx])
        
if __name__=="__main__":
    env=Env()
    agent=MC_agent()
    total_episode=10000
    sr=0
    acc1=[]
    
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
            state=next_state
            action=agent.get_action(state)
            next_action=agent.get_action(next_state)
            total_reward+=reward

            
            if done:
                if episode%100==0:
                    print('finished at ', state)
                    print('episode: {}, The number of step: {}\n The sequence of action is {} \n The total reward is: {}'.format(episode,walk,action_sequence,total_reward))
                    
                if state==env.goal:
                    sr+=1
                    acc1.append(sr/total_episode*100)
                    
                agent.update()
                agent.memory.clear()
                break
print('total episode: ',total_episode) 
print('sr : ',sr)     
print('The accuary: ',sr/total_episode*100,'%')

plt.figure(1)
plt.title('MC_rein ACC')                      
plt.plot(acc1,marker='.',c='y')
plt.grid()                                
plt.show()
