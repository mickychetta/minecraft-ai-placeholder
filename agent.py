# Rllib docs: https://docs.ray.io/en/latest/rllib.html

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint

import gym
import ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo


class ResourceCollector(gym.Env):

    def __init__(self, env_config):
        # Static Parameters
        self.size = 10
        self.reward_density = .1
        self.penalty_density = .02
        self.obs_size = 5
        self.max_global_steps = (self.size * 2) ** 2
        self.log_frequency = 10
        self.action_dict = {
            0: 'move 1',  # Move one block forward
            1: 'turn 1',  # Turn 90 degrees to the right
            2: 'turn -1',  # Turn 90 degrees to the left
            3: 'attack 1',  # Destroy block
            4: 'jumpmove 1'  # Jump up and move forward 1 block
        }
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

        # Rllib Parameters
        self.action_space = Discrete(len(self.action_dict))
        self.observation_space = Box(-1, 6, shape=(
            np.prod([2, self.obs_size, self.obs_size]), ), dtype=np.int32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # ResourceCollector Parameters
        self.obs = None
        self.obsdict = None # Stores last json loaded observation
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.resources_collected = {
            "diamond": [0],
            "redstone": [0],
            "coal": [0],
            "emerald": [0],
            "iron_ore": [0],
            "gold_ore": [0]
        }
        self.steps = []
        self.episode_start = time.time()
        self.episode_end = time.time()

    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        # Get amount of resources collected
        if self.episode_step > 0:
            resrcs = set(self.resources_collected.keys())
            for i in range(0,39):
                key = 'InventorySlot_'+str(i)+'_item'
                if key in self.obsdict:
                    item = self.obsdict[key] 
                    if item in self.resources_collected:
                        self.resources_collected[item].append(int(self.obsdict[u'InventorySlot_'+str(i)+'_size']))
                        resrcs.remove(item)
                if len(resrcs) == 0:
                    break

            # Add 0 for resources not found
            for r in resrcs:
                self.resources_collected[r].append(0)
        
        # Reset Malmo
        world_state = self.init_malmo()

        # Reset Variables
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        print("Starting episode", len(self.steps))
        self.episode_return = 0
        self.episode_step = 0
        self.episode_start = time.time()
        self.episode_end = time.time()

        # Log
        if len(self.returns) > self.log_frequency and \
                len(self.returns) % self.log_frequency == 0:
            self.log_returns()
            self.log_resources_collected()

        # Get Observation
        self.obs = self.get_observation(world_state)

        return self.obs.flatten()

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <int> index of the action to take

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """

        # Get Action
        command = self.action_dict[action]
        # allow if there is a block we want to mine in the ground in front of the agent (y=1)
        allow_break_action = self.obs[0, int(
            self.obs_size/2)-1, int(self.obs_size/2)] > 0
        if command != 'attack 1' or allow_break_action:
            self.agent_host.sendCommand(command)
            time.sleep(.1)
            self.episode_step += 1
            self.episode_end = time.time()

        # Get Done
        # Done is true if we reach time limit
        done = False
        if self.episode_end - self.episode_start >= 30.0:
            done = True
            time.sleep(2)
        # Done is also true if lava is stepped into
        elif (self.obs[0, int(self.obs_size/2)-1, int(self.obs_size/2)] == -1 and (command == 'move 1' or command == 'jumpmove 1')):
            done = True
            time.sleep(2)

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs = self.get_observation(world_state)

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
        self.episode_return += reward
        #print("Current reward:", self.episode_return)

        return self.obs.flatten(), reward, done, dict()

    def get_mission_xml(self):
        xml = ""

        # Ores
        for _ in range(int(self.max_global_steps * 0.25)):
            x = randint(-self.size, self.size)
            z = randint(-self.size, self.size)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='redstone_ore' />".format(
                x, z)

        for _ in range(int(self.max_global_steps * 0.17)):
            x = randint(-self.size, self.size)
            z = randint(-self.size, self.size)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='coal_ore' />".format(
                x, z)

        for _ in range(int(self.max_global_steps * 0.13)):
            x = randint(-self.size, self.size)
            z = randint(-self.size, self.size)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='emerald_ore' />".format(
                x, z)

        for _ in range(int(self.max_global_steps * 0.1)):
            x = randint(-self.size, self.size)
            z = randint(-self.size, self.size)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='iron_ore' />".format(
                x, z)

        for _ in range(int(self.max_global_steps * 0.07)):
            x = randint(-self.size, self.size)
            z = randint(-self.size, self.size)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='gold_ore' />".format(
                x, z)

        for _ in range(int(self.max_global_steps * 0.02)):
            x = randint(-self.size, self.size)
            z = randint(-self.size, self.size)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='diamond_ore' />".format(
                x, z)

        # Lava
        for _ in range(int(self.max_global_steps * 0.007)):
            x = randint(-self.size, self.size)
            z = randint(-self.size, self.size)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='lava' />".format(
                x-1, z+1)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='lava' />".format(
                x, z+1)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='lava' />".format(
                x+1, z+1)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='lava' />".format(
                x-1, z)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='lava' />".format(
                x, z)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='lava' />".format(
                x-1, z-1)

        for _ in range(int(self.max_global_steps * 0.005)):
            x = randint(-self.size, self.size)
            z = randint(-self.size, self.size)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='lava' />".format(
                x-1, z+1)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='lava' />".format(
                x, z+1)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='lava' />".format(
                x+1, z+1)

        for _ in range(int(self.max_global_steps * 0.005)):
            x = randint(-self.size, self.size)
            z = randint(-self.size, self.size)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='lava' />".format(
                x, z+1)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='lava' />".format(
                x-1, z)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='lava' />".format(
                x+1, z)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='lava' />".format(
                x, z-1)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='lava' />".format(
                x, z-2)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='lava' />".format(
                x, z)

        for _ in range(int(self.max_global_steps * 0.01)):
            x = randint(-self.size, self.size)
            z = randint(-self.size, self.size)
            xml += "<DrawBlock x='{}'  y='1' z='{}' type='lava' />".format(
                x, z)

        # Air
        for _ in range(int(self.max_global_steps * 0.09)):
            x = randint(-self.size, self.size)
            z = randint(-self.size, self.size)
            xml += "<DrawBlock x='{}' y='1' z='{}' type='air' />".format(x, z)

        # Rail
        for _ in range(int(self.max_global_steps * 0.1)):
            x = randint(-self.size, self.size)
            z = randint(-self.size, self.size)
            xml += "<DrawBlock x='{}' y='2' z='{}' type='rail' />".format(x, z)

        # Gravel
        for _ in range(int(self.max_global_steps * 0.3)):
            x = randint(-self.size, self.size)
            z = randint(-self.size, self.size)
            xml += "<DrawBlock x='{}' y='1' z='{}' type='gravel' />".format(
                x, z)

        return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                    <About>
                        <Summary>Resource Gatherer</Summary>
                    </About>

                    <ServerSection>
                        <ServerInitialConditions>
                            <Time>
                                <StartTime>12000</StartTime>
                                <AllowPassageOfTime>true</AllowPassageOfTime>
                            </Time>
                            <Weather>clear</Weather>
                        </ServerInitialConditions>
                        <ServerHandlers>
                            <FlatWorldGenerator generatorString="3;7,2;21;"/>
                            <DrawingDecorator>''' + \
            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='air'/>".format(-self.size, self.size, -self.size, self.size) + \
            "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='stone'/>".format(-self.size, self.size, -self.size, self.size) + \
            "<DrawCuboid x1='{}' x2='{}' y1='0' y2='-5' z1='{}' z2='{}' type='bedrock'/>".format(-self.size, self.size, -self.size, self.size) + \
            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='stone'/>".format(-self.size-1, self.size, -self.size-1, -self.size-1) + \
            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='stone'/>".format(-self.size-1, self.size, self.size, self.size) + \
            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='stone'/>".format(-self.size-1, -self.size-1, -self.size-1, self.size) + \
            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='2' z1='{}' z2='{}' type='stone'/>".format(self.size, self.size, -self.size-1, self.size) + \
            "<DrawCuboid x1='{}' x2='{}' y1='3' y2='6' z1='{}' z2='{}' type='stone'/>".format(-self.size-1, self.size, -self.size-1, -self.size-1) + \
            "<DrawCuboid x1='{}' x2='{}' y1='3' y2='6' z1='{}' z2='{}' type='stone'/>".format(-self.size-1, self.size, self.size, self.size) + \
            "<DrawCuboid x1='{}' x2='{}' y1='3' y2='6' z1='{}' z2='{}' type='stone'/>".format(-self.size-1, -self.size-1, -self.size-1, self.size) + \
            "<DrawCuboid x1='{}' x2='{}' y1='3' y2='6' z1='{}' z2='{}' type='stone'/>".format(self.size, self.size, -self.size-1, self.size) + \
            xml + \
            '''<DrawBlock x='0'  y='2' z='0' type='air' />
                                <DrawBlock x='0'  y='1' z='0' type='stone' />
                            </DrawingDecorator>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>

                    <AgentSection mode="Survival">
                        <Name>SpeedMiner</Name>
                        <AgentStart>
                            <Placement x="0.5" y="2" z="0.5" pitch="60" yaw="0"/>
                            <Inventory>
                                <InventoryItem slot="0" type="diamond_pickaxe"/>
                            </Inventory>
                        </AgentStart>
                        <AgentHandlers>
                            <DiscreteMovementCommands/>
                            <ObservationFromFullInventory/>
                            <ObservationFromFullStats/>
                            <ObservationFromGrid>
                                <Grid name="floorAll">
                                    <min x="-'''+str(int(self.obs_size/2))+'''" y="-1" z="-'''+str(int(self.obs_size/2))+'''"/>
                                    <max x="'''+str(int(self.obs_size/2))+'''" y="0" z="'''+str(int(self.obs_size/2))+'''"/>
                                </Grid>
                            </ObservationFromGrid>
                            <RewardForCollectingItem>
                                <Item type="diamond" reward="6"/> 
                                <Item type="gold_ore" reward="5"/> 
                                <Item type="iron_ore" reward="4"/> 
                                <Item type="emerald" reward="3"/> 
                                <Item type="coal" reward="2"/> 
                                <Item type="redstone" reward="0.1"/> 
                            </RewardForCollectingItem>
                            <RewardForTouchingBlockType>
                                <Block type="bedrock" reward="-1" />
                                <Block type="lava" reward="-5" />
                                <Block type="flowing_lava" reward="-5" />
                            </RewardForTouchingBlockType>
                            <AgentQuitFromTimeUp timeLimitMs="'''+str(30000)+'''" />
                            <AgentQuitFromTouchingBlockType>
                                <Block type="lava" />
                                <Block type="flowing_lava" />
                            </AgentQuitFromTouchingBlockType>
                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''

    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.get_mission_xml(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        # add Minecraft machines here as available
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))

        for retry in range(max_retries):
            try:
                self.agent_host.startMission(
                    my_mission, my_clients, my_mission_record, 0, 'ResourceCollector')
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)

        return world_state

    def get_observation(self, world_state):
        """
        Use the agent observation API to get a 2 x 5 x 5 grid around the agent. 
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array>
        """
        obs = np.zeros((2, self.obs_size, self.obs_size))

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)
                self.obsdict = observations

                # Get observation
                grid = observations['floorAll']
                grid_translated = [self.blocks_dict[x]
                                   if x in self.blocks_dict else 0 for x in grid]
                obs = np.reshape(
                    grid_translated, (2, self.obs_size, self.obs_size))

                # Rotate observation with orientation of agent
                yaw = observations['Yaw']
                if yaw == 270:
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif yaw == 0:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif yaw == 90:
                    obs = np.rot90(obs, k=3, axes=(1, 2))

                break

        return obs

    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns, box, mode='same')
        plt.clf()
        plt.plot(self.steps, returns_smooth)
        plt.title('Episode Reward Return')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps, self.returns):
                f.write("{}\t{}\n".format(step, value))

    def log_resources_collected(self):
        """
        Log the current resources collected as a graph and text file
        """
        plt.clf()
        fig, axs = plt.subplots(3, 2, sharex=True)
        fig.suptitle('Resources Mined By Agent')
        plt.rc('font', size=8)
        
        box = np.ones(self.log_frequency) / self.log_frequency
        
        resrc_smooth = np.convolve(self.resources_collected['diamond'], box, mode='same')
        axs[0][0].plot(self.steps, resrc_smooth, color='skyblue')
        axs[0][0].set_title('Diamond')

        resrc_smooth = np.convolve(self.resources_collected['gold_ore'], box, mode='same')
        axs[0][1].plot(self.steps, resrc_smooth, color='gold')
        axs[0][1].set_title('Gold Ore')

        resrc_smooth = np.convolve(self.resources_collected['iron_ore'], box, mode='same')
        axs[1][0].plot(self.steps, resrc_smooth, color='gray')
        axs[1][0].set_title('Iron Ore')

        resrc_smooth = np.convolve(self.resources_collected['emerald'], box, mode='same')
        axs[1][1].plot(self.steps, resrc_smooth, color='mediumaquamarine')
        axs[1][1].set_title('Emerald')

        resrc_smooth = np.convolve(self.resources_collected['coal'], box, mode='same')
        axs[2][0].plot(self.steps, resrc_smooth, color='black')
        axs[2][0].set_title('Coal')

        resrc_smooth = np.convolve(self.resources_collected['redstone'], box, mode='same')
        axs[2][1].plot(self.steps, resrc_smooth, color='firebrick')
        axs[2][1].set_title('Redstone')

        for ax in axs.flat:
            ax.set(xlabel='Steps', ylabel='Amt Mined Per Episode')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        plt.tight_layout()

        plt.savefig('resources.png')


if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=ResourceCollector, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    while True:
        print(trainer.train())
