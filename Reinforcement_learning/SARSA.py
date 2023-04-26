import numpy as np

class Env:
    def __init__(self):
        self.grid_width = 10
        self.grid_height = self.grid_width
        self.action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)] # U, D, L, R
        self.obstacles=[[1,4],[1,5],[1,6],[2,3],[2,5],[2,7],[4,4],[4,6],[8,8],[9,8]]
        self.goal = [5,5]
        
    def step(self, state, action):
        x, y = state
        
        # get next state by action
        x+= action[0]
        y+= action[1]
        
        if x < 0 :
            x = 0
        elif x > (self.grid_width-1) :
            x = (self.grid_width-1)

        if y < 0 :
            y = 0
        elif y > (self.grid_width-1) :
            y = (self.grid_width-1)
        
        next_state = [x, y]
        
        if next_state in self.obstacles:
            reward = -1
            done = True
        elif next_state == self.goal:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        
        return next_state, reward, done
    
    def reset(self):
        return [0, 0]
    
class SARSA_agent:
    def __init__(self):
        self.action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_text= ['U', 'D', 'L', 'R']
        self.grid_width = 10
        self.grid_height = self.grid_width
        self.Qtable = np.zeros((self.grid_width, self.grid_height, len(self.action_grid)))
        self.e = .1
        self.learning_rate = .01
        self.discount_factor = .95
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
    total_episode = 10000
    sarsa_sr = 0
    
    for episode in range(total_episode):
        action_sequence=[]
        total_reward = 0
        walk = 0
        
        # initial state, action, done
        state = env.reset()
        action = agent.get_action(state)
        done = False
        
        while not done:  
            agent.save_actionseq(action_sequence, action)
            
            # next state, action
            next_state, reward, done = env.step(state, action)
            next_action = agent.get_action(next_state)

            # update Qtable
            agent.update(state, action, reward, next_state, next_action)
            
            total_reward += reward
            
            if done:
                if episode % 100 == 0:
                    print('finished at', next_state)
                    print('episode :{}, The number of step:{}\n The sequence of action is:\
                          {}\nThe total reward is: {}\n'.format(episode, walk, action_sequence, total_reward))
                    
                if state == env.goal:
                    sarsa_sr += 1

            state = next_state
            action = agent.get_action(state)
            walk += 1

        # update sarsa_sr if goal is reached
        if state == env.goal:
            sarsa_sr += 1
            
    print('total episode: ',total_episode) 
    print('sarsa_sr : ',sarsa_sr)           
    print('The accuracy :', sarsa_sr/total_episode*100, '%')
