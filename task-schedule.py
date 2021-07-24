"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import copy
from itertools import combinations
import run
import random

mu = 0
sigma = 0.12

class TaskScheduleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num	Observation               Min             Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                -24 deg         24 deg
        3	Pole Velocity At Tip      -Inf            Inf
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average reward is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):


        self.task_pool = None
        self.cpu = None
        self.arrived_task = None
        # 建立action的字典
        self.action_table = {}

        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        for i in range(len(x)):
            self.action_table['%s' % i] = (x[i],)

        a = list(combinations(x, 2))

        for i in range(len(a)):
            self.action_table['%s' % (i + len(x))] = a[i]

        b = list(combinations(x, 3))

        for i in range(len(b)):
            self.action_table['%s' % (i + len(x) + len(a))] = b[i]

        c = list(combinations(x, 4))

        for i in range(len(c)):
            self.action_table['%s' % (i + len(x) + len(a) + len(b))] = c[i]



        self.action_space = spaces.Discrete(385)


        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):


        # 设置完成标志位为false
        done = False

        # 根据action，采取相应行动
        process_task_id = self.action_table['%s' % str(action)]
        # 如果选择的action中有全0任务，state不做更改，reward = -100
        action_correct_flag = True
        for item in process_task_id:
            if self.task_pool[item][5] < 0:
                action_correct_flag = False

        # 如果选中的action中执行的任务数量大于空闲CPU个数，reward = -100
        idle_CPU = 0
        for task in self.cpu:
            if task[1] == 0:
                idle_CPU += 1
        if len(process_task_id) != idle_CPU:
            action_correct_flag = False

        # 当action_correct_flag为真时，才继续下一步判断，否则直接结束该轮step
        if action_correct_flag == True:

            # 准备计算获得的utility
            reward = 0
            # action执行的任务数量等于空闲cpu的数量，将action代表的任务插入空闲cpu，state改变
            for item in process_task_id:
                for i in range(len(self.cpu)):
                    if self.cpu[i][5] == 0:
                        # 将task pool中选定的任务插入cpu中，
                        self.cpu[i] = copy.deepcopy(self.task_pool[item])
                        self.task_pool[item] = [-1+random.gauss(mu,sigma),-1+random.gauss(mu,sigma),-1+random.gauss(mu,sigma),-1+random.gauss(mu,sigma),-1+random.gauss(mu,sigma),-1+random.gauss(mu,sigma)]
                        # 计算这个被执行的任务现在的utility
                        for task in self.arrived_task:
                            if task[0] == self.cpu[i][0]:
                                wait_time = task[2] - self.cpu[i][2]
                                reward += run.compute_utility(task[3], wait_time, task[1], task[2], task[5])
                                break
                        break

            # 处理所有cpu中的任务，直到有cpu变为空闲，计算是否有任务miss deadline
            # 找到cpu队列中的最短任务
            shortest_task = 999999
            for item in self.cpu:
                if item[1] < shortest_task and item[1] != 0:
                    shortest_task = item[1]
            # 所有task pool中的任务都等待
            for item in self.task_pool:
                if item[5] > 0:
                    item[2] -= shortest_task
            # cpu队列中的任务都进行处理
            for i in range(len(self.cpu)):
                self.cpu[i][1] -= shortest_task
                if self.cpu[i][1] == 0:
                    self.cpu[i] = [0,0,0,0,0,0]
            # 判断当前状态下的cpu、task pool组合是否可行
            self.task_pool_test = copy.deepcopy(self.task_pool)
            delets_list = []
            for i in range(len(self.task_pool)):
                if self.task_pool_test[i][5] < 0:
                    delets_list.append(i)
            self.task_pool_test = np.delete(self.task_pool_test, delets_list, 0)
            feasible_flag = run.global_admission_control(self.task_pool_test, 5, self.cpu)
            # feasible flag=0，当前组合可行；反之，不可行，给定reward
            if feasible_flag == 0:
                pass
            else:
                reward = -100

            # 判断当前状态是否完成调度，若已完成调度，计算当前剩余的utility
            # 计算空闲cpu数量
            idle_cpu = 0
            for item in self.cpu:
                if item[1] == 0:
                    idle_cpu += 1
            # 计算task pool剩余几个任务
            task_number = 0
            for item in self.task_pool:
                if item[5] > 0:
                    task_number += 1
            # 空闲cpu数量大于等于task pool长度，true；反之，false
            if feasible_flag == 0:
                done = bool(idle_cpu >= task_number)

            if done:
                reward += run.compute_current_total_utility_of_taskpool(self.task_pool_test, self.arrived_task)

        else:
            reward = -100

        # 根据action_correct_flag更新state
        if action_correct_flag == True:
            while len(self.task_pool) < 10:
                self.task_pool = np.vstack((self.task_pool, [-1+random.gauss(mu,sigma),-1+random.gauss(mu,sigma),-1+random.gauss(mu,sigma),-1+random.gauss(mu,sigma),-1+random.gauss(mu,sigma),-1+random.gauss(mu,sigma)]))
            self.state = np.vstack((self.task_pool, self.cpu))
            # 将task pool中所有任务的原始信息加入state
            for task in self.task_pool:
                for item in self.arrived_task:
                    if task[0] == item[0]:
                        self.state = np.vstack((self.state, item))
            while len(self.state) < 25:
                self.state = np.vstack((self.state, [-1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma)]))

        return self.state, reward, done, {}

    def reset(self):
        while len(self.task_pool) < 10:
            self.task_pool = np.vstack((self.task_pool, [-1+random.gauss(mu,sigma),-1+random.gauss(mu,sigma),-1+random.gauss(mu,sigma),-1+random.gauss(mu,sigma),-1+random.gauss(mu,sigma),-1+random.gauss(mu,sigma)]))

        self.state = np.vstack((self.task_pool, self.cpu))
        # 将task pool中所有任务的原始信息加入state
        for task in self.task_pool:
            for item in self.arrived_task:
                if task[0] == item[0]:
                    self.state = np.vstack((self.state, item))
        while len(self.state) < 25:
            self.state = np.vstack((self.state, [-1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma)]))
        self.steps_beyond_done = None
        return self.state

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
