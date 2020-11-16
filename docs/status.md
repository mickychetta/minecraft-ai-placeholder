---
layout: default
title: Status
---

# Project Summary
Our goals for our project have changed since our initial proposal. We have decided to change our goal from building a nether portal to a efficient and fast resource gatherer. Our goal for our resource gatherer is to efficiently gather resources within a certain time limit. If time permits, we plan to have our agent craft weapons and armors as well. 

# Approach
We are taking a reinforcement learning approach using rllib for our resource gatherer. Observations, actions, rewards, and terminal states are important to identify for our agent, and below, we have briefly described them. 

## Observations
Our observations are given to us within the `ObservationsFromGrid` tag in the `get_mission_xml()` function and the `get_observation()` function. Our observations are given to us in the form of a multi dimensional numpy array. Similar to the 2 x 5 x 5 observation grid surrounding the agent in assignment 2, we will also have an observation grid. However, our observation grid will be of the entire map, and does not change as the agent moves. 

## Actions
Our agent's actions will consist of continuous movements, including turning left, turning right, moving forward, jumping, and attacking. This will be represented by our action dictionary.

## Rewards
Our agent will be rewarded for mining a variety of materials, including: diamond, gold, iron, emerald, coal, lapis lazuli, redstone. Higher valued ores will be more scarce within our grid.


| Materials | Rewards | Density |
| ----------- | ----------- | ----------- |
| Diamond | 6 | 3% |
| Gold | 5 | 6% |
| Iron | 4 | 10% |
| Emerald | 3 | 13% |
| Coal | 2 | 16% |
| Lapis Lazuli | 1 | 20% |
| Redstone | 1 | 20% |

<br>

![ores](./images/ores.jpg)

## Terminal States
For our terminal states, we have deided to go with a timed approach rather than a step based approach. We set a threshold of 30 seconds for our agent to efficiently collect resources. Alternatively, if all resources have been collected in our grid, this also represents a terminal state.

# Evaluation
In our evaluation process, we will take into consideration the rewards given from mining each ore. Each ore gives a reward based on its scarcity, which has been predetermined by our own choosing. Throughout our training period, we want our agent to mine higher valued rewards rather than the lower valued rewards due to our given time constraints. 

# Remaining Goals and Challenges
For the next 4-5 weeks, we 

# Resources Used
rllib
numpy
matplotlib
stackoverflow
XML documentation
markdown styling
campuswire
github


# goals for tmr
- mvp for algorithm
- learn rllib (gym env)
- observation grid xml stationary grid
- change terminal state to 30 sec time limie
- agent movement torwards coordinates 
- finish status report/video