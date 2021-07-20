[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FGhaiyur%2FRl-Agents&count_bg=%23000000&title_bg=%23000000&icon=probot.svg&icon_color=%23E7E7E7&title=Bots+Visited&edge_flat=false)](https://hits.seeyoufarm.com)
# [Rl-Agents](https://github.com/Ghaiyur/Rl-Agents) 

## SIRL - Space Invader RL

Q-learning is a simple yet quite powerful algorithm to create a cheat sheet for our agent. This helps the agent figure out exactly which action to perform.

But what if this cheatsheet is too long? Imagine an environment with 10,000 states and 1,000 actions per state. This would create a table of 10 million cells. Things will quickly get out of control!

It is pretty clear that we can’t infer the Q-value of new states from already explored states. This presents two problems:

First, the amount of memory required to save and update that table would increase as the number of states increases
Second, the amount of time required to explore each state to create the required Q-table would be unrealistic

### Model Summary 

![ms1](https://user-images.githubusercontent.com/26713317/125749429-85c9e514-9613-4d01-9a12-d9082f46487a.png)

## BO - Break Out RL

A longstanding goal of artificial intelligence is the development of algorithms capable of
general competency in a variety of tasks and domains without the need for domain-specific
tailoring. To this end, different theoretical frameworks have been proposed to formalize the
notion of “big” artificial intelligence (e.g., Russell, 1997; Hutter, 2005; Legg, 2008). Similar
ideas have been developed around the theme of lifelong learning: learning a reusable, highlevel understanding of the world from raw sensory data (Thrun & Mitchell, 1995; Pierce &
Kuipers, 1997; Stober & Kuipers, 2008; Sutton et al., 2011). The growing interest in competitions such as the General Game Playing competition (Genesereth, Love, & Pell, 2005),
Reinforcement Learning competition (Whiteson, Tanner, & White, 2010), and the International Planning competition (Coles et al., 2012) also suggests the artificial intelligence
community’s desire for the emergence of algorithms that provide general competency.

### Model Summary 

![ms1](https://user-images.githubusercontent.com/26713317/125749429-85c9e514-9613-4d01-9a12-d9082f46487a.png)

## Autonomous Taxi - Numpy Q-learning from Scratch 

Q-Table is just a fancy name for a simple lookup table where we calculate the maximum expected future rewards for action at each state. Basically, this table will guide us to the best action at each state.

Each Q-table score will be the maximum expected future reward that the robot will get if it takes that action at that state. This is an iterative process, as we need to improve the Q-Table at each iteration.

But the questions are:

- How do we calculate the values of the Q-table?
- Are the values available or predefined?

![qalgo](https://user-images.githubusercontent.com/26713317/126094661-af71fc21-16d6-4a2e-9ed9-33bec996c66b.png)

## FlappyDQN - Deeper Q- Learning

So, what are the steps involved in reinforcement learning using deep Q-learning networks (DQNs)?

All the past experience is stored by the user in memory
The next action is determined by the maximum output of the Q-network
The loss function here is mean squared error of the predicted Q-value and the target Q-value – Q*. This is basically a regression problem. However, we do not know the target or actual value here as we are dealing with a reinforcement learning problem. Going back to the Q-value update equation derived fromthe Bellman equation. we have:
![image](https://user-images.githubusercontent.com/26713317/126251741-a68ee1f9-fb73-45ea-be86-5772ba946835.png)

The section in green represents the target. We can argue that it is predicting its own value, but since R is the unbiased true reward, the network is going to update its gradient using backpropagation to finally converge.

## Model Summary 

![dqn](https://user-images.githubusercontent.com/26713317/126251622-afd3dbc9-372e-4f79-a8fb-45fa5723891f.png)


