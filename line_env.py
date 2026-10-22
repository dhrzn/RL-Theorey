# line_env.py
# Toy RL environment: a cart (self.x) on a number line and a flag (self.t) as the target.
# Plain Python, no installs. Step 1 of my plan, before turning it into a Gymnasium class.
# The cart gets pushed by an action, and the reward is the negative distance to the flag.
# Running this file directly plays 100 random pushes as a baseline agent.

import random

class LineEnv:
    def __init__(self): #defining default values
        self.x = 0
        self.t = 0

    def reset(self): #this returns an observation  because this defines the starting position and goal (basically describes the environment)
        random_x = random.uniform(-10,10) #this is what the random point agent starts with
        random_t = random.uniform(-10,10) #this is the target goal
        self.x = random_x
        self.t = random_t
        return [self.x , self.t]

    def step(self, action_step_size): #this is the action input because based on the observation that is given, it will decide to do an action. Based on that action it will recieve a score/reward determining how good said action was. it will observe the score and the environment again to decide the next action 
        first_guess = self.x  #defining again what the agent starts with
        new_position = first_guess + action_step_size #note: action_step_size is the action because this is the value that is coming in
        self.x = new_position
        error = self.x - self.t  #calculating error in order to determine rewards
        reward_points = -abs(error) #having the highest point achievable to be zero [0 = best any number lower is worse] 
        done = reward_points > -0.1
        # prediction: x=3, t=8, step(0.6) returns ([3.6, 8], -4.4, False) because the distance 4.4 is not under 0.1
        return [self.x, self.t], reward_points, done #the observation is [self.x, self.t] and is defined by the action that was taken (the result of action) | and the reward = the score that is spat back out after the action (reward_points)


# the if __name__ line below is a switch for the test code only.
# running "python line_env.py" directly: __name__ is "__main__", so the test loop runs.
# importing it from another file (import line_env): the class loads, but the indented test loop is skipped.
# the lines above the if always run in both cases.
if __name__ == "__main__":
    env = LineEnv() #env builds the "LineEnv" class blue print with its own self.x and own things. basically all of the functions now live under env
    env.reset()
    for i in range(100):
        obs, reward, done = env.step(random.uniform(-1, 1)) #obs, reward, done catches the three things step() hands back and assigning them a name
        print(obs, reward, done)
        if done: #if done: means "if done is true"
            break #breaks amnd leaves the for loop


#note to self: agents can only interpert "highest number is best" or "lowest number is best" it can never follow "zero is the best" or "10 is the best" directly.