---
layout: default
title: Final Report
---

## Video

## Project Summary
Our goals for our project have largely remained the same since our status report. Our main goal is for our Minecraft agent to gather resources as efficiently as possible within a certain time frame. These resources are a set of ores, which will be later defined in our approach section. Our agent is placed into a simulated cave environment with these resources and obstacles, such as lava and rails. The video above provides a clearer idea of what this environment looks like.

Solving this problem of mining as many resources within a time frame may seem simple and using AI/ML to tackle this problem may seem excessive. In our peer review feedback, Professor Singh mentions that "just gathering resources is not that impressive since the agent can just learn to break as many blocks as possible". To solve this issue, we decided to penalize the agent in certain ways, such as penalizing the agent when it touches bedrock. Bedrock, in our environment, is found under the layer of resources and stone, so if too many blocks are broken, than the agent would be penalized. This is implemented in order for our agent to avoid falling into holes. We also weigh rewards for ores differently and set their distribution density accordingly, allowing for the agent to learn which ores to prioritize over others.

<br>

## Approaches
In our baseline approach, we used a `6x6 `grid with only ores on the floor. The disadvantage of this approach is that our agent would always mine ores since our observation space always allowed the break action. When our agent fell into a hole, he was stuck there indefinitely because `jumpmove` had not been incorporated into our action space. This limited our agent from training effectively since we did not add any negative rewards to optimize the best mining. Also, there was only one graph of resources collected which didn't give any specific metrics to the different ores. 

The new approach we decided on was to penalize our agent for falling into lava, flowing lava, and bedrock. In this way, our agent will be trained to avoid these obstacles as they are mining valuable ores efficiently in our cave environment. The environment we used was a `20x20` grid with different types of ores and block types to simulate a real cave environment. With the additional `jumpmove` action, our agent is able to escape holes unlike our previous baseline approach. In addition, we added rails, gravel, and fixed the density of our rewards. The new rewards table is shown in the `Reward` section. Since adding lava, we added falling into lava as an additional terminal state. This will reset our agent and create accurate metrics and training data. We created 6 additional graphs, each representing the collection of a different ore versus total steps taken; these graphs are constructed in function `log_resources_collected()`. These additional graphs provide more insight into the performance of our agent. We continued to use the RLlib library with different reinforcement learning algorithms: PPO and DQN. We decided to compare the different algorithms to see which resulted in better performance.

### PPO
<img src="https://spinningup.openai.com/en/latest/_images/math/99621d5bcaccd056d6ca3aeb48a27bf8cc0e640c.svg" alt="loss function" width="650">

<br>

### DQN
<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/1_lTVHyzT3d26Bd_znaKaylQ.png" alt="loss function" width="650">
<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-2019-04-16-at-5.46.01-PM.png" alt="loss function" width="650">


<br>


### Observations
Our observations are given to us within the `ObservationsFromGrid` tag in the `get_mission_xml()` function and the `get_observation()` function. Our observations are given to us in the form of a multi dimensional numpy array. Our observation grid represents the 2 x 5 x 5 area surrounding the agent. The observation space that we feed to the trainer is made to differentiate between the different types of ores, as well as blocks we want to avoid like lava. We use these numerical values to represent the different blocks in the observation space:
```
self.blocks_dict = {
            "redstone_ore": 1,
            "coal_ore": 2,
            "emerald_ore": 3,
            "iron_ore": 4,
            "gold_ore": 5,
            "diamond_ore": 6,
            "lava": -1,
            "flowing_lava": -1
        }
```
Any other block is represented as a 0.

<br>


### Actions
Our agent's actions will consist of discrete movements, including turning left, turning right, moving forward, jumping and moving forward, and attacking. This will be represented by our action dictionary.

<br>


### Rewards
Our agent will be rewarded for mining a variety of materials, including: diamond, gold, iron, emerald, coal, redstone. Higher valued ores will be more scarce within our grid. It should be noted that mining some ores results in collecting more than 1 of the associated resource. For example, mining 1 redstone ore block results in 5 redstones collected. The rewards below indicate the reward for each individual resource collected.


| Materials | Rewards | Density |
| ----------- | ----------- | ----------- |
| Diamond | 6 | 2% |
| Gold | 5 | 7% |
| Iron | 4 | 10% |
| Emerald | 3 | 13% |
| Coal | 2 | 17% |
| Redstone | 0.1 | 25% |
| Lava | -5 | 10% |

<br>

![ores](./images/ores.png)

### Terminal States
For our terminal states, we have decided to go with a timed approach rather than a step based approach. We set a threshold of 30 seconds for our agent to efficiently collect resources. Additionally, if the agent dies by touching lava, then the mission will end as well.

## Evaluation

## References

- https://microsoft.github.io/malmo/0.14.0/Schemas/Types.html
- https://microsoft.github.io/malmo/0.14.0/Schemas/MissionHandlers.html
- https://docs.ray.io/en/master/rllib.html