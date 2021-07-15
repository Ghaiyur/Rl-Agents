[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FGhaiyur%2FRl-Agents&count_bg=%23000000&title_bg=%23000000&icon=probot.svg&icon_color=%23E7E7E7&title=Bots+Visited&edge_flat=false)](https://hits.seeyoufarm.com)
# Rl-Agents 

## SIRL - Space Invader RL

Q-learning is a simple yet quite powerful algorithm to create a cheat sheet for our agent. This helps the agent figure out exactly which action to perform.

But what if this cheatsheet is too long? Imagine an environment with 10,000 states and 1,000 actions per state. This would create a table of 10 million cells. Things will quickly get out of control!

It is pretty clear that we can’t infer the Q-value of new states from already explored states. This presents two problems:

First, the amount of memory required to save and update that table would increase as the number of states increases
Second, the amount of time required to explore each state to create the required Q-table would be unrealistic

### Model Summary 

![ms1](https://user-images.githubusercontent.com/26713317/125749429-85c9e514-9613-4d01-9a12-d9082f46487a.png)



