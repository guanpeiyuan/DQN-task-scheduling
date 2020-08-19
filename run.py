from __future__ import division
import numpy as np
import random
from itertools import combinations
import math
import time
import copy
import pandas as pd
import threading
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import gc
import gym
from sklearn import preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

# 全局变量，统计训练成功的次数
cycles = 0

def creat_taskgroup(task_number, computation_expectation, slack_expectation):
    task_list = np.empty((task_number, 6))
    task_id = []
    for i in range(task_number):
        task_id.append(i)
    computation = np.random.poisson(computation_expectation, task_number)
    slack = np.random.poisson(slack_expectation, task_number)
    utility_function = np.random.randint(3, size=task_number)

    for i in range(task_number):
        task_list[i][0] = task_id[i]
        task_list[i][1] = computation[i]
        if task_list[i][1] == 0:
            task_list[i][1] = 1
        task_list[i][2] = slack[i]
        task_list[i][3] = utility_function[i]


    return task_list


def creat_task(task_id, computation_expectation, slack_expectation, timeslot, utility_type):
    task_list = []
    computation = np.random.randint(computation_expectation - 4, computation_expectation + 5)
    slack = np.random.randint(slack_expectation - 19, slack_expectation + 20)
    utility_function = np.random.randint(1, 4)
    utility_coefficient = 1
    task_list.append(task_id)
    task_list.append(computation)
    task_list.append(slack)
    task_list.append(utility_function)
    task_list.append(timeslot)
    task_list.append(utility_coefficient)

    return task_list


def utility_exponential(wait_time, computation, utility_coefficient, slack):
    utility = utility_coefficient * computation * math.exp((computation / slack) * (-wait_time))
    return utility


def utility_log(wait_time, computation, utility_coefficient):
    utility = (-1) * math.log(wait_time + 1) + computation * utility_coefficient
    return utility


def utility_line(wait_time, computation, slack, utility_coefficient):
    if slack == 0:
        utility = computation * utility_coefficient
    else:
        utility = ((-1) * utility_coefficient * computation / slack) * wait_time + utility_coefficient * computation

    return utility


def utility_step(wait_time, computation, slack, utility_coefficient):
    if wait_time < slack / 3:
        utility = computation * utility_coefficient

    if slack / 3 <= wait_time < 2 * slack / 3:
        utility = computation * utility_coefficient / 2

    if 2 * slack / 3 <= wait_time:
        utility = 0

    return utility


def compute_utility(utility_function, wait_time, computation, slack, utility_coefficient):
    utility = 0

    if utility_function == 0:
        utility = utility_log(wait_time, computation, utility_coefficient)

    if utility_function == 1:
        utility = utility_step(wait_time, computation, slack, utility_coefficient)

    if utility_function == 2:
        utility = utility_line(wait_time, computation, slack, utility_coefficient)

    if utility_function == 3:
        utility = utility_exponential(wait_time, computation, utility_coefficient, slack)

    return utility


def compute_current_total_utility_of_taskpool(task_list, task_accept_list):
    task_pool = copy.deepcopy(task_list)

    current_utility_list = []

    for i in range(len(task_pool)):

        for original_task in task_accept_list:

            if original_task[0] == task_pool[i][0]:
                task_computation_original = original_task[1]

                task_slack_original = original_task[2]

                task_utility_original = original_task[3]

                task_utility_coefficient = original_task[5]

        current_utility_list.append(
            compute_utility(task_utility_original, task_slack_original - task_pool[i][2], task_computation_original,
                            task_slack_original, task_utility_coefficient))

    current_total_utility = sum(current_utility_list)

    return current_total_utility


def utility_reserved_if_taski_executed(task_list, task_accept_list, task_id):
    task_pool = copy.deepcopy(task_list)

    remain_utility = 0

    # calculate utility of task with task_id

    for item in task_pool:

        if item[0] == task_id:
            current_computation = item[1]

            break

    # update status of other tasks

    for i in range(len(task_pool)):

        if task_pool[i][0] != task_id:
            task_pool[i][2] = task_pool[i][2] - current_computation

    # delete task i

    for i in range(len(task_pool)):

        if task_pool[i][0] == task_id:
            task_pool = np.delete(task_pool, i, 0)

            break

    # calculate total utility of other tasks

    remain_utility += compute_current_total_utility_of_taskpool(task_pool, task_accept_list)

    return remain_utility


def global_utility_reserved_if_taskset_exexuted(task_list, task_accept_list):
    task_pool = copy.deepcopy(task_list)

    remaining_utility = 0

    for task in task_pool:

        for item in task_accept_list:

            if task[0] == item[0]:
                remaining_utility += compute_utility(item[3], item[0] - task[0], item[1], item[2], item[5])

    return remaining_utility


def admission_control(task_ndarray_need_judge, current_executing_computation):
    flag = 0
    task_execution = 0

    task_list = copy.deepcopy(task_ndarray_need_judge)

    for task in task_list:

        task[2] -= current_executing_computation

        if task[2] < 0:
            flag = 1

            break

    task_in_sort = sort_by_deadline(task_list)

    for i in range(len(task_in_sort)):
        task_execution += task_in_sort[i][1]

        task_deadline = task_in_sort[i][1] + task_in_sort[i][2]

        if task_execution <= task_deadline:
            pass
        else:
            flag = 1

    return flag


def global_admission_control(task_list, core_number, executing_task_list):
    task_list_test = copy.deepcopy(task_list)

    executing_task_list_test = copy.deepcopy(executing_task_list)

    task_list_test = sort_by_deadline(task_list_test)

    # 0 means task list is feasible, 1 means not feasible

    admission_control_flag = 0

    # check the feasibility when task list is not empty
    # if the task list is empty, no need to check feasibility

    while len(task_list_test) > 0:

        # check the task pool, if there is a negative slack

        for item in task_list_test:

            if item[2] < 0:
                admission_control_flag = 1

                break

        # check the executing task list, if there is a negative slack

        for item in executing_task_list_test:

            if item[2] < 0:
                admission_control_flag = 1

                break

        # is there empty core? if yes, use task pool to fill in,

        for i in range(core_number):

            # the ith core is empty

            if executing_task_list_test[i][1] == 0:

                if len(task_list_test) > 0:
                    executing_task_list_test[i] = task_list_test[0]

                    task_list_test = np.delete(task_list_test, 0, 0)

        # by the end of this part, fill in is over

        # for this part, check the executing procedure

        # after fill in, the task pool is empty, it is feasible

        if len(task_list_test) == 0:

            pass

        # after fill in, the task pool is not empty, check further

        else:

            # process one time slot, update cores and task pool

            for i in range(len(executing_task_list_test)):

                executing_task_list_test[i][1] -= 1

                # if the executing task is finished, set core to idle

                if executing_task_list_test[i][1] == 0:
                    executing_task_list_test[i] = [0, 0, 0, 0, 0, 0]

            for task in task_list_test:
                task[2] -= 1

        if admission_control_flag == 1:
            break

    return admission_control_flag


def sort_by_id(task_list):
    target_list = copy.deepcopy(task_list)
    n = len(target_list)

    for i in range(n - 1):
        count = 0
        for j in range(0, n - 1 - i):
            if target_list[j][0] > target_list[j + 1][0]:
                target_list[[j, j + 1], :] = target_list[[j + 1, j], :]
                count += 1
        if 0 == count:
            break

    return target_list


def sort_by_deadline(task_list):
    target_list = copy.deepcopy(task_list)
    n = len(target_list)

    for i in range(n - 1):
        count = 0
        for j in range(0, n - 1 - i):
            if target_list[j][1] + target_list[j][2] > target_list[j + 1][1] + target_list[j + 1][2]:
                target_list[[j, j + 1], :] = target_list[[j + 1, j], :]
                count += 1

            if target_list[j][1] + target_list[j][2] == target_list[j + 1][1] + target_list[j + 1][2]:

                if target_list[j][1] > target_list[j + 1][1]:
                    target_list[[j, j + 1], :] = target_list[[j + 1, j], :]

        if 0 == count:
            break
    return target_list


def sort_by_computation(task_list):
    target_list = copy.deepcopy(task_list)
    n = len(target_list)

    for i in range(n - 1):
        count = 0
        for j in range(0, n - 1 - i):
            if target_list[j][1] > target_list[j + 1][1]:
                target_list[[j, j + 1], :] = target_list[[j + 1, j], :]
                count += 1
        if 0 == count:
            break
    return target_list


def sort_by_average_utility(task_list, task_accept_list):
    target_list = copy.deepcopy(task_list)
    n = len(target_list)

    for i in range(n - 1):
        count = 0
        for j in range(0, n - 1 - i):

            for item in task_accept_list:
                if item[0] == target_list[j][0]:
                    task_j_slack = item[2]
                    task_j_computation = item[1]
                if item[0] == target_list[j + 1][0]:
                    task_j1_slack = item[2]
                    task_j1_computation = item[1]

            if compute_utility(target_list[j][3], task_j_slack - target_list[j][2], task_j_computation, task_j_slack,
                               target_list[j][5]) / \
                    target_list[j][1] < compute_utility(target_list[j + 1][3], task_j1_slack - target_list[j + 1][2],
                                                        task_j1_computation, task_j1_slack, target_list[j + 1][5]) / \
                    target_list[j + 1][1]:
                target_list[[j, j + 1], :] = target_list[[j + 1, j], :]
                count += 1
        if 0 == count:
            break

    task_list_sorted = target_list

    return task_list_sorted


def sort_by_least_utility_lost(task_list, task_accept_list):
    target_list = copy.deepcopy(task_list)

    n = len(target_list)

    current_total_utility = compute_current_total_utility_of_taskpool(target_list, task_accept_list)

    utility_list = []

    for i in range(len(target_list)):

        for j in range(len(task_accept_list)):

            if target_list[i][0] == task_accept_list[j][0]:
                utility_list.append(compute_utility(task_accept_list[j][3], task_accept_list[j][2] - target_list[i][2],
                                                    task_accept_list[j][1], task_accept_list[j][2],
                                                    task_accept_list[j][5]))

    target_list = np.insert(target_list, 6, utility_list, 1)

    utility_reserved_list = []

    for i in range(len(target_list)):
        utility_reserved_list.append(
            utility_reserved_if_taski_executed(target_list, task_accept_list, target_list[i][0]))

    target_list = np.insert(target_list, 7, utility_reserved_list, 1)

    # rest_time = []
    #
    # for i in range(len(target_list)):
    #
    #     rest_time.append(count_rest_time_one_task_pool(target_list,target_list[i][0]))

    for i in range(n - 1):
        count = 0
        for j in range(0, n - 1 - i):

            j_value = (current_total_utility - target_list[j][7]) / target_list[j][1]

            j1_value = (current_total_utility - target_list[j + 1][7]) / target_list[j + 1][1]

            if j_value > j1_value:
                target_list[[j, j + 1], :] = target_list[[j + 1, j], :]

            if j_value == j1_value:
                if target_list[j][1] > target_list[j + 1][1]:
                    target_list[[j, j + 1], :] = target_list[[j + 1, j], :]

                # rest_time[j],rest_time[j+1] = rest_time[j+1],rest_time[j]

                # if utility_reserved_if_taski_executed(target_list, task_accept_list,target_list[j][0]) / count_rest_time_one_task_pool(target_list,target_list[j][0]) < utility_reserved_if_taski_executed(target_list, task_accept_list, target_list[j + 1][0]) / count_rest_time_one_task_pool(target_list, target_list[j + 1][0]):
                #     target_list[[j, j + 1], :] = target_list[[j + 1, j], :]

                # if compute_utility(target_list[j][3], task_j_slack - target_list[j][2], task_j_computation, task_j_slack)/target_list[j][1] == compute_utility(target_list[j+1][3], task_j1_slack - target_list[j+1][2],task_j1_computation,task_j1_slack)/target_list[j+1][1]:
                #
                #     if target_list[j][1] > target_list[j+1][1]:
                #
                #         target_list[[j, j + 1], :] = target_list[[j + 1, j], :]

                count += 1
        if 0 == count:
            break

    target_list = np.delete(target_list, 7, 1)

    target_list = np.delete(target_list, 6, 1)

    task_list_sorted = target_list

    return task_list_sorted


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table initial values
        columns=actions,  # actions's name
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table, ACTIONS, EPSILON):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:  # act greedy
        action_name = state_actions.idxmax()  # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(S, A, N_STATES):
    # This is how agent will interact with the environment
    execute_task_id = int(A)

    # This is how agent will interact with the environment
    if A == 'right':  # move right
        if S == N_STATES - 2:  # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:  # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def rl(task_list, task_arrive_list):
    target_list = copy.deepcopy(task_list)
    search_depth = 5
    if len(target_list) <= 5:
        N_STATES = len(target_list)  # the length of the 1 dimensional world
    else:
        N_STATES = search_depth
    ACTIONS = []  # available actions
    for i in range(len(target_list)):
        ACTIONS.append(str(target_list[i][0]))
    EPSILON = 0.9  # greedy police
    ALPHA = 0.1  # learning rate
    GAMMA = 0.9  # discount factor
    MAX_EPISODES = 13  # maximum episodes

    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        current_depth = 0
        step_counter = 0
        S = 0

        while current_depth != search_depth:

            A = choose_action(S, q_table, ACTIONS, EPSILON)
            S_, R = get_env_feedback(S, A, N_STATES)  # take action & get next state and reward
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()  # next state is not terminal
            else:
                q_target = R  # next state is terminal
                is_terminated = True  # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            step_counter += 1
    return q_table


def sort_by_UPT(task_list, task_accept_list):
    target_list = copy.deepcopy(task_list)
    n = len(target_list)

    for i in range(n - 1):
        count = 0
        for j in range(0, n - 1 - i):

            for item in task_accept_list:
                if item[0] == target_list[j][0]:
                    task_j_coefficient = item[5]
                    task_j_computation = item[1]
                if item[0] == target_list[j + 1][0]:
                    task_ji_coefficient = item[5]
                    task_j1_computation = item[1]

            if task_j_coefficient < task_ji_coefficient:
                target_list[[j, j + 1], :] = target_list[[j + 1, j], :]
                count += 1

            if task_j_coefficient == task_ji_coefficient and task_j_computation > task_j1_computation:
                target_list[[j, j + 1], :] = target_list[[j + 1, j], :]

        if 0 == count:
            break

    task_list_sorted = target_list

    return task_list_sorted


def sort_by_DUPT(task_list, task_accept_list):
    target_list = copy.deepcopy(task_list)
    n = len(target_list)

    for i in range(n - 1):
        count = 0
        for j in range(0, n - 1 - i):

            for item in task_accept_list:
                if item[0] == target_list[j][0]:
                    wait_time_j = item[2] - target_list[j][2]
                    computation_j = item[1]
                    slack_j = item[2]

                if item[0] == target_list[j + 1][0]:
                    wait_time_j1 = item[2] - target_list[j + 1][2]
                    computation_j1 = item[1]
                    slack_j1 = item[2]

            if compute_utility(target_list[j][4], wait_time_j, computation_j, slack_j, target_list[j][5]) / \
                    target_list[j][1] < compute_utility(target_list[j + 1][4], wait_time_j1, computation_j1, slack_j1,
                                                        target_list[j + 1][5]) / target_list[j + 1][1]:
                target_list[[j, j + 1], :] = target_list[[j + 1, j], :]
                count += 1

        if 0 == count:
            break

    task_list_sorted = target_list

    return task_list_sorted


def sort_by_PLA(task_list, task_accept_list):
    target_list = copy.deepcopy(task_list)

    obtain = []

    for i in range(len(target_list)):

        for j in range(len(task_accept_list)):

            if target_list[i][0] == task_accept_list[j][0]:
                obtain.append(task_accept_list[j][5])

                break

    task_id_list = []

    for i in range(len(target_list)):
        task_id_list.append(target_list[i][0])

    potential_lost_list = []

    for i in range(len(target_list)):

        potential_lost = 0

        for j in range(len(task_accept_list)):

            if task_accept_list[j][0] in task_id_list and task_accept_list[j][0] != target_list[i][0]:
                potential_lost += task_accept_list[j][1] * task_accept_list[j][5] / task_accept_list[j][2]

        potential_lost_list.append(potential_lost)

    n = len(target_list)

    for i in range(n - 1):
        count = 0
        for j in range(0, n - 1 - i):

            if 2 * obtain[j] - potential_lost_list[j] < 2 * obtain[j + 1] - potential_lost_list[j + 1]:
                target_list[[j, j + 1], :] = target_list[[j + 1, j], :]

                count += 1

        if 0 == count:
            break

    task_list_sorted = target_list

    return task_list_sorted


def sort_by_X(task_list, task_accept_list):
    target_list = copy.deepcopy(task_list)

    # creat a metric to store the utility loss rate information
    task_utility_loss_list = np.empty((0, 4))

    for task in target_list:

        task_utility_loss_information = []

        for item in task_accept_list:

            if item[0] == task[0]:
                wait_time = item[2] - task[2]

                task_utility_loss_information.append(task[0])

                task_utility_loss_information.append(task[1])

                task_utility_loss_information.append(
                    compute_utility(item[3], wait_time, item[1], item[2], item[5]) / task[2])

                task_utility_loss_information.append(0)

        task_utility_loss_list = np.insert(task_utility_loss_list, len(task_utility_loss_list),
                                           task_utility_loss_information, 0)

    for i in range(len(task_utility_loss_list)):

        for j in range(len(task_utility_loss_list)):

            if j != i:
                task_utility_loss_list[i][3] += task_utility_loss_list[j][2]

    n = len(target_list)

    for i in range(n - 1):
        count = 0

        for j in range(0, n - 1 - i):

            if task_utility_loss_list[j][1] * task_utility_loss_list[j][3] > task_utility_loss_list[j + 1][1] * \
                    task_utility_loss_list[j + 1][3]:
                task_utility_loss_list[[j, j + 1], :] = task_utility_loss_list[[j + 1, j], :]

                target_list[[j, j + 1], :] = target_list[[j + 1, j], :]

                count += 1

        if 0 == count:
            break
    return target_list


def global_PLA(task_arrive_list, total_timeslot, core_number, times, max_lenth_of_task_pool):
    task_pool = np.empty((0, 6))

    execute_sequence = []

    utility_sequence = []

    task_wait_time = 0

    task_arrive_list_sorted = sort_by_id(task_arrive_list)

    task_decline_number_EDF = 0

    task_decline_information = np.zeros(5)

    task_decline_id = []

    accumulate_utility = 0

    utility_detail = []

    task_pool_length = []

    executing_task = [0, 0, 0, 0, 0, 0]

    executing_task_list = []

    for i in range(core_number):
        executing_task_list.append(executing_task)

    error_flag = 0

    # system runs total time slots

    for timeslot in range(total_timeslot):

        print('PLA, {}th time, time slot {}'.format(times, timeslot))

        if (timeslot + 1) == 100:
            task_decline_information[0] += task_decline_number_EDF

        if (timeslot + 1) == 200:
            task_decline_information[1] += task_decline_number_EDF

        if (timeslot + 1) == 300:
            task_decline_information[2] += task_decline_number_EDF

        if (timeslot + 1) == 400:
            task_decline_information[3] += task_decline_number_EDF

        if (timeslot + 1) == 500:
            task_decline_information[4] += task_decline_number_EDF

        # for each time slot, find if there are tasks coming

        for i in range(len(task_arrive_list_sorted)):

            # if there are tasks coming, try to insert the tasks into task pool

            # global part

            if task_arrive_list_sorted[i][4] == timeslot:

                # if task pool is not full

                if len(task_pool) < max_lenth_of_task_pool:

                    task_pool = np.insert(task_pool, len(task_pool), task_arrive_list_sorted[i], 0)

                    # use admission control to jude if the task if acceptable

                    if global_admission_control(task_pool, core_number, executing_task_list) == 0:

                        accumulate_utility += task_arrive_list_sorted[i][1] * task_arrive_list_sorted[i][5]

                    # task i is not acceptable for the current task pool

                    else:

                        task_decline_number_EDF += 1

                        task_decline_id.append(task_arrive_list_sorted[i][0])

                        # delete task i

                        for j in range(len(task_pool)):

                            if task_pool[j][0] == task_arrive_list_sorted[i][0]:
                                task_pool = np.delete(task_pool, j, 0)

                                break

                else:

                    task_decline_number_EDF += 1

                    task_decline_id.append(task_arrive_list_sorted[i][0])

            if task_arrive_list_sorted[i][4] > timeslot:
                break

            # global part over

        # update all the cores, if a core is empty

        # task_pool = sort_by_least_utility_lost(task_pool, task_arrive_list_sorted)
        #
        # for i in range(core_number):
        #
        #     if executing_task_list[i][1] == 0:
        #
        #         if len(task_pool) > 0:
        #
        #             for j in range(len(task_pool)):
        #
        #                 test_pool = copy.deepcopy(task_pool)
        #
        #                 executing_task_list[i] = test_pool[j]
        #
        #                 test_pool = np.delete(test_pool, j, 0)
        #
        #                 if global_admission_control(test_pool, core_number, executing_task_list) == 0:
        #
        #                     task_pool = np.delete(task_pool, j, 0)
        #
        #                     break
        #
        #                 else:
        #
        #                     executing_task_list[i] = [0,0,0,0,0,0]

        # update all the cores, if some cores are empty
        idle_cpu = 0
        for task in executing_task_list:
            if task[1] == 0:
                idle_cpu += 1
        # there is idle cpu
        if idle_cpu > 0:
            # task pool is not empty
            if len(task_pool) > 0:
                # tasks in pool is more than idle cpu, need to choose a combination to execute
                if len(task_pool) > idle_cpu:
                    # find all the combinations of tasks
                    candidate_combinations = list(combinations(task_pool, idle_cpu))

                    potential_lost_rate = 999

                    # for each combination, find the temporary_lost_rate
                    for combination in candidate_combinations:

                        test_pool = copy.deepcopy(task_pool)

                        executing_task_list_test = copy.deepcopy(executing_task_list)

                        # one idle cpu
                        if len(combination) == 1:
                            # utility of current task pool
                            utility_with_combination = compute_current_total_utility_of_taskpool(test_pool,
                                                                                                 task_arrive_list_sorted)

                            # fill task into executing_task_list_test
                            for task in combination:

                                for i in range(len(executing_task_list_test)):

                                    if executing_task_list_test[i][1] == 0:
                                        executing_task_list_test[i] = copy.deepcopy(task)

                                        break

                            # find the shortest computation in executing task list
                            shortest_computation = 999

                            for i in range(core_number):

                                if executing_task_list_test[i][1] < shortest_computation:
                                    shortest_computation = executing_task_list_test[i][1]

                            # other tasks in task pool wait
                            for i in range(len(test_pool)):

                                if test_pool[i][0] != combination[0][0]:
                                    test_pool[i][2] -= shortest_computation

                            # utility of updated task pool
                            utility_without_combination = compute_current_total_utility_of_taskpool(test_pool,
                                                                                                    task_arrive_list_sorted)

                            # delete task from test pool
                            for i in range(len(test_pool)):

                                if test_pool[i][0] == combination[0][0]:
                                    test_pool = np.delete(test_pool, i, 0)

                                    break

                            # update the status of all cpu
                            for task in executing_task_list_test:
                                task[1] -= shortest_computation

                            # compute the potential lost rate for this combination
                            temporary_lost_rate = (
                                                              utility_with_combination - utility_without_combination) / shortest_computation

                            if temporary_lost_rate < potential_lost_rate and global_admission_control(test_pool,
                                                                                                      core_number,
                                                                                                      executing_task_list_test) == 0:
                                potential_lost_rate = temporary_lost_rate

                                print('temporary_lost_rate:{}'.format(temporary_lost_rate))

                                target_combination = combination

                        # more than one idle cpu
                        if len(combination) > 1:
                            # utility of current task pool
                            utility_with_combination = compute_current_total_utility_of_taskpool(test_pool,
                                                                                                 task_arrive_list_sorted)

                            # fill task into executing_task_list_test
                            for task in combination:

                                for i in range(len(executing_task_list_test)):

                                    if executing_task_list_test[i][1] == 0:
                                        executing_task_list_test[i] = copy.deepcopy(task)

                                        break

                            # find the shortest computation in executing task list
                            shortest_computation = 999

                            for i in range(core_number):

                                if executing_task_list_test[i][1] < shortest_computation:
                                    shortest_computation = executing_task_list_test[i][1]

                            # other tasks in task pool wait
                            task_set = []
                            for task in combination:
                                task_set.append(task[0])

                            for i in range(len(test_pool)):

                                if test_pool[i][0] not in task_set:
                                    test_pool[i][2] -= shortest_computation

                            # utility of updated task pool
                            utility_without_combination = compute_current_total_utility_of_taskpool(test_pool,
                                                                                                    task_arrive_list_sorted)

                            # delete task from test pool
                            temporary_delete = []

                            for task in combination:

                                for i in range(len(test_pool)):

                                    if test_pool[i][0] == task[0]:
                                        temporary_delete.append(i)

                            test_pool = np.delete(test_pool, temporary_delete, 0)

                            # update the status of all cpu
                            for task in executing_task_list_test:
                                task[1] -= shortest_computation

                            # compute the potential lost rate for this combination
                            temporary_lost_rate = (
                                                              utility_with_combination - utility_without_combination) / shortest_computation

                            if temporary_lost_rate < potential_lost_rate and global_admission_control(test_pool,
                                                                                                      core_number,
                                                                                                      executing_task_list_test) == 0:
                                potential_lost_rate = temporary_lost_rate

                                print('temporary_lost_rate:{}'.format(temporary_lost_rate))

                                target_combination = combination

                    print('target_combination:{}'.format(target_combination))

                    for task in target_combination:

                        for i in range(len(executing_task_list)):

                            if executing_task_list[i][1] == 0:
                                executing_task_list[i] = task

                                break

                        temporary_delete = []
                        for i in range(len(task_pool)):

                            if task_pool[i][0] == task[0]:
                                temporary_delete.append(i)

                        task_pool = np.delete(task_pool, temporary_delete, 0)

                # tasks in pool is less than or equal to idle cpu, fill tasks in cpu
                else:
                    for i in range(len(task_pool)):

                        for j in range(core_number):

                            if executing_task_list[j][1] == 0:
                                executing_task_list[j] = task_pool[i]

                                break

                    task_pool = np.empty((0, 6))

            # task pool is empty
            else:
                pass
        # there is no idle cpu
        else:
            pass

        # all task pools and cores are updated, process the tasks now

        for j in range(core_number):

            # there is a task executing in core i

            if executing_task_list[j][1] > 0:

                # task in the core i is processed for 1 unit

                executing_task_list[j][1] -= 1

                # if the executing task is finished

                if executing_task_list[j][1] == 0:
                    # add the task into executed sequence

                    execute_sequence.append(executing_task_list[j][0])

                    # compute the utility of executing task

                    for task in task_arrive_list:

                        if task[0] == executing_task_list[j][0]:
                            utility = compute_utility(task[3], task[2] - executing_task_list[j][2], task[1],
                                                      task[2], task[5])

                            utility_sequence.append(utility)

                            task_wait_time += task[2] - executing_task_list[j][2]

                            break

                    # set executing task list i to empty, means core i need to find a new task to process

                    executing_task_list[j] = [0, 0, 0, 0, 0, 0]

        # all tasks in task pool wait 1 unit

        for task in task_pool:
            task[2] -= 1

        for task in task_pool:

            if task[2] < 0:
                error_flag = 1

                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        for task in executing_task_list:

            if task[2] < 0:
                error_flag = 1

                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        if error_flag == 1:
            break

        print('executing task:')

        for item in executing_task_list:
            print(item)

        print('task pool:')

        print(task_pool)

        print('')

        task_pool_length.append(len(task_pool))

        utility_detail.append(sum(utility_sequence))

    print('')

    print('the execute sequence of PLA is:{}'.format(execute_sequence.__len__()))

    print('')

    print('the accumulate utility is {}'.format(accumulate_utility))

    print('')

    print('the utility of PLA is:{}'.format(sum(utility_sequence)))

    print('')

    return len(execute_sequence), accumulate_utility, utility_sequence, task_decline_information, error_flag, task_wait_time, utility_detail, task_pool_length


def global_DRL(task_arrive_list, total_timeslot, core_number, times, max_lenth_of_task_pool):
    task_pool = np.empty((0, 6))

    execute_sequence = []

    utility_sequence = []

    task_wait_time = 0

    task_arrive_list_sorted = sort_by_id(task_arrive_list)

    task_decline_number_EDF = 0

    task_decline_information = np.zeros(5)

    task_decline_id = []

    accumulate_utility = 0

    utility_detail = []

    task_pool_length = []

    executing_task = [0, 0, 0, 0, 0, 0]

    executing_task_list = []

    for i in range(core_number):
        executing_task_list.append(executing_task)

    error_flag = 0

    # system runs total time slots

    for timeslot in range(total_timeslot):

        print('delta, {}th time, time slot {}'.format(times, timeslot))

        if (timeslot + 1) == 100:
            task_decline_information[0] += task_decline_number_EDF

        if (timeslot + 1) == 200:
            task_decline_information[1] += task_decline_number_EDF

        if (timeslot + 1) == 300:
            task_decline_information[2] += task_decline_number_EDF

        if (timeslot + 1) == 400:
            task_decline_information[3] += task_decline_number_EDF

        if (timeslot + 1) == 500:
            task_decline_information[4] += task_decline_number_EDF

        # for each time slot, find if there are tasks coming

        for i in range(len(task_arrive_list_sorted)):

            # if there are tasks coming, try to insert the tasks into task pool

            # global part

            if task_arrive_list_sorted[i][4] == timeslot:

                # if task pool is not full

                if len(task_pool) < max_lenth_of_task_pool:

                    task_pool = np.insert(task_pool, len(task_pool), task_arrive_list_sorted[i], 0)

                    # use admission control to jude if the task if acceptable

                    if global_admission_control(task_pool, core_number, executing_task_list) == 0:

                        accumulate_utility += task_arrive_list_sorted[i][1] * task_arrive_list_sorted[i][5]

                    # task i is not acceptable for the current task pool

                    else:

                        task_decline_number_EDF += 1

                        task_decline_id.append(task_arrive_list_sorted[i][0])

                        # delete task i

                        for j in range(len(task_pool)):

                            if task_pool[j][0] == task_arrive_list_sorted[i][0]:
                                task_pool = np.delete(task_pool, j, 0)

                                break

                else:

                    task_decline_number_EDF += 1

                    task_decline_id.append(task_arrive_list_sorted[i][0])

            if task_arrive_list_sorted[i][4] > timeslot:
                break

            # global part over

        print("cpu information:")
        for item in executing_task_list:
            print(item)
        print('')
        print("task pool:")
        print(task_pool)
        print('')


        """
        DRL running part
        """

        # update all the cores, if a core is empty
        # how many idle cpu? how long is the task pool?
        # counting idle cpu number
        idle_cpu = 0
        for item in executing_task_list:
            if item[0] == 0:
                idle_cpu += 1
        # if idle cpu is more than length of task pool, just fill task in
        if idle_cpu >= len(task_pool):
            for i in range(len(task_pool)):
                for j in range(len(executing_task_list)):
                    if executing_task_list[j][1] == 0:
                        executing_task_list[j] = copy.deepcopy(task_pool[i])
                        break
            # all tasks in task pool should be deleted
            task_pool = np.empty((0,6))

        elif idle_cpu == 0:
            pass
        # if idle cpu is less or equal to length of task pool, use model to schedule
        else:
            # prepare the state
            task_pool_copy = copy.deepcopy(task_pool)
            while len(task_pool_copy) < 10:
                task_pool_copy = np.vstack((task_pool_copy, [-1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma)]))
            current_state = np.vstack((task_pool_copy, executing_task_list))
            # 将task pool中所有任务的原始信息加入state
            for task in task_pool_copy:
                for item in task_arrive_list:
                    if task[0] == item[0]:
                        current_state = np.vstack((current_state, item))
            while len(current_state) < 25:
                current_state = np.vstack((current_state, [-1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma)]))

            # current state已经准备好，查找所有可行解

            # 根据idle cpu确定action的采样范围
            action_range = []
            if idle_cpu == 1:
                for m in range(10):
                    action_range.append(m)
            elif idle_cpu == 2:
                for m in range(10, 55):
                    action_range.append(m)
            elif idle_cpu == 3:
                for m in range(55, 175):
                    action_range.append(m)
            else:
                for m in range(175, 385):
                    action_range.append(m)

            # 对采样范围内的每一种process task id的组合进行演算
            reward_max = -999
            action_candidat = ()
            # 当前状态下预测的reward
            reward_current = agent.model.predict(preprocessing.scale(current_state[:,1:5]).reshape((1,25,4,1)))[0][0]


            for i in range(len(action_range)):
                # 进入循环，准备验证集
                current_state_test = copy.deepcopy(current_state)
                # 当前组合的reward设置为0
                reward_step = 0

                # 对第i个方案进行验证，是否选到了空任务
                process_task_id = agent.action_table['%s' % str(action_range[i])]
                negative_task_number = 0
                for id in process_task_id:
                    if current_state_test[id][5] < 0:
                        negative_task_number += 1
                # 根据negative_task_number，当没有negative_task时继续下一步，否则跳过该组合
                if negative_task_number == 0:

                    # 判断当前组合是否会导致miss deadline,如果不会，则计算这一步能拿到的真实reward
                    for item in process_task_id:
                        for j in range(10, 15):
                            # 第j个cpu空闲，将第item个任务插入，第item个任务置空
                            if current_state_test[j][5] == 0:
                                current_state_test[j] = copy.deepcopy(current_state_test[item])
                                current_state_test[item] = [-1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma), -1+random.gauss(mu,sigma)]
                                # 计算这个被执行的任务现在的utility
                                for task_0 in task_arrive_list:
                                    if task_0[0] == current_state_test[j][0]:
                                        wait_time = task_0[2] - current_state_test[j][2]
                                        reward_step += compute_utility(task_0[3], wait_time, task_0[1], task_0[2], task_0[5])
                                        break
                                break
                    # 对current task test中的任务池信息进行处理，使得


                    # 找到cpu队列中的最短任务
                    shortest_task = 999999
                    for i_1 in range(10, 15):
                        if current_state_test[i_1][1] < shortest_task and current_state_test[i_1][1] != 0:
                            shortest_task = current_state_test[i_1][1]

                    # 所有task pool中的任务都等待
                    delete_list = []
                    for i_2 in range(0, 10):
                        if current_state_test[i_2][5] > 0:
                            current_state_test[i_2][2] -= shortest_task

                    # cpu队列中的任务都进行处理
                    for i_3 in range(10, 15):
                        current_state_test[i_3][1] -= shortest_task
                        if current_state_test[i_3][1] == 0:
                            current_state_test[i_3] = [0, 0, 0, 0, 0, 0]

                    delets_list = []
                    for i in range(0, 10):
                        if current_state_test[i][5] < 0:
                            delets_list.append(i)
                    temperary_list = np.delete(current_state_test[0:10, :], delets_list, 0)

                    # 对可行性进行判断
                    feasible_flag = global_admission_control(temperary_list, 5, current_state_test[10:15, :])

                    # feasible_flag等于0，当前组合可行，加上下一个state的预测值；反之，跳过该组合
                    if feasible_flag == 0:
                        avavilabe_cpu = 0
                        for item in current_state_test[10:15,:]:
                            if item[5] < 0:
                                avavilabe_cpu += 1

                        waiting_task = 0
                        for item in current_state_test[0:10,:]:
                            if item[5] > 0:
                                waiting_task += 1
                        if avavilabe_cpu < waiting_task:

                            reward_future = agent.model.predict(preprocessing.scale(current_state_test[:,1:5]).reshape((1,25,4,1)))[0][0]

                        else:
                            reward_future = compute_current_total_utility_of_taskpool(current_state_test[0:waiting_task,:],task_arrive_list_sorted)
                        reward_candidate = reward_step + reward_future
                        print("当前组合的reward为：{}".format(reward_candidate))

                        # 寻找使得reward最大的任务组合
                        if reward_candidate >= reward_max:
                            reward_max = copy.deepcopy(reward_candidate)

                            # 设置候选action
                            action_candidat = copy.deepcopy(process_task_id)
                            print("候选任务组合：{}".format(action_candidat))

                    else:
                        pass


                else:
                    pass


            # 最终方案确定，按照方案执行
            print("")
            print("选择执行的任务是：{}".format(action_candidat))
            print("")

            delete_list = []

            for id in action_candidat:
                for j in range(5):
                    # 第j个cpu空闲，将第item个任务插入，第item个任务置空
                    if executing_task_list[j][5] == 0:
                        executing_task_list[j] = copy.deepcopy(task_pool[id])
                        delete_list.append(id)
                        break
            task_pool = np.delete(task_pool, delete_list, 0)

        """
        DRL running part over
        """

        print("information after scheduling:")

        print("cpu information:")
        for item in executing_task_list:
            print(item)
        print('')
        print("task pool:")
        print(task_pool)
        print('')
        print('')
        print('')

        # all task pools and cores are updated, process the tasks now
        for j in range(core_number):

            # there is a task executing in core i

            if executing_task_list[j][1] > 0:

                # task in the core i is processed for 1 unit

                executing_task_list[j][1] -= 1

                # if the executing task is finished

                if executing_task_list[j][1] == 0:
                    # add the task into executed sequence

                    execute_sequence.append(executing_task_list[j][0])

                    # compute the utility of executing task

                    for task in task_arrive_list:

                        if task[0] == executing_task_list[j][0]:
                            utility = compute_utility(task[3], task[2] - executing_task_list[j][2], task[1],
                                                      task[2], task[5])

                            utility_sequence.append(utility)

                            task_wait_time += task[2] - executing_task_list[j][2]

                            break

                    # set executing task list i to empty, means core i need to find a new task to process

                    executing_task_list[j] = [0, 0, 0, 0, 0, 0]

        # all tasks in task pool wait 1 unit

        for task in task_pool:
            task[2] -= 1

        for task in task_pool:

            if task[2] < 0:
                error_flag = 1

                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        for task in executing_task_list:

            if task[2] < 0:
                error_flag = 1

                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        if error_flag == 1:
            break

        task_pool_length.append(len(task_pool))

        utility_detail.append(sum(utility_sequence))

    print('')

    print('the execute sequence of delta is:{}'.format(execute_sequence.__len__()))

    print('')

    print('the accumulate utility is {}'.format(accumulate_utility))

    print('')

    print('the utility of delta is:{}'.format(sum(utility_sequence)))

    print('')

    return len(execute_sequence), accumulate_utility, utility_sequence, task_decline_information, error_flag, task_wait_time, utility_detail, task_pool_length


def global_X(task_arrive_list, total_timeslot, core_number, times, max_lenth_of_task_pool):
    task_pool = np.empty((0, 6))

    execute_sequence = []

    utility_sequence = []

    task_wait_time = 0

    task_arrive_list_sorted = sort_by_id(task_arrive_list)

    task_decline_number_EDF = 0

    task_decline_information = np.zeros(5)

    task_decline_id = []

    accumulate_utility = 0

    utility_detail = []

    task_pool_length = []

    executing_task = [0, 0, 0, 0, 0, 0]

    executing_task_list = []

    for i in range(core_number):
        executing_task_list.append(executing_task)

    error_flag = 0

    # system runs total time slots

    for timeslot in range(total_timeslot):

        print('X, {}th time, time slot {}'.format(times, timeslot))

        if (timeslot + 1) == 100:
            task_decline_information[0] += task_decline_number_EDF

        if (timeslot + 1) == 200:
            task_decline_information[1] += task_decline_number_EDF

        if (timeslot + 1) == 300:
            task_decline_information[2] += task_decline_number_EDF

        if (timeslot + 1) == 400:
            task_decline_information[3] += task_decline_number_EDF

        if (timeslot + 1) == 500:
            task_decline_information[4] += task_decline_number_EDF

        # for each time slot, find if there are tasks coming

        for i in range(len(task_arrive_list_sorted)):

            # if there are tasks coming, try to insert the tasks into task pool

            # global part

            if task_arrive_list_sorted[i][4] == timeslot:

                # if task pool is not full

                if len(task_pool) < max_lenth_of_task_pool:

                    task_pool = np.insert(task_pool, len(task_pool), task_arrive_list_sorted[i], 0)

                    # use admission control to jude if the task if acceptable

                    if global_admission_control(task_pool, core_number, executing_task_list) == 0:

                        accumulate_utility += task_arrive_list_sorted[i][1] * task_arrive_list_sorted[i][5]

                    # task i is not acceptable for the current task pool

                    else:

                        task_decline_number_EDF += 1

                        task_decline_id.append(task_arrive_list_sorted[i][0])

                        # delete task i

                        for j in range(len(task_pool)):

                            if task_pool[j][0] == task_arrive_list_sorted[i][0]:
                                task_pool = np.delete(task_pool, j, 0)

                                break

                else:

                    task_decline_number_EDF += 1

                    task_decline_id.append(task_arrive_list_sorted[i][0])

            if task_arrive_list_sorted[i][4] > timeslot:
                break

            # global part over

        # update all the cores, if some cores are empty

        # sort task pool

        task_pool = sort_by_X(task_pool, task_arrive_list)

        for i in range(core_number):

            if executing_task_list[i][1] == 0:

                if len(task_pool) > 0:

                    for j in range(len(task_pool)):

                        test_pool = copy.deepcopy(task_pool)

                        executing_task_list[i] = test_pool[j]

                        test_pool = np.delete(test_pool, j, 0)

                        if global_admission_control(test_pool, core_number, executing_task_list) == 0:

                            task_pool = np.delete(task_pool, j, 0)

                            break

                        else:

                            executing_task_list[i] = [0, 0, 0, 0, 0, 0]

        # all task pools and cores are updated, process the tasks now

        for j in range(core_number):

            # there is a task executing in core i

            if executing_task_list[j][1] > 0:

                # task in the core i is processed for 1 unit

                executing_task_list[j][1] -= 1

                # if the executing task is finished

                if executing_task_list[j][1] == 0:
                    # add the task into executed sequence

                    execute_sequence.append(executing_task_list[j][0])

                    # compute the utility of executing task

                    for task in task_arrive_list:

                        if task[0] == executing_task_list[j][0]:
                            utility = compute_utility(task[3], task[2] - executing_task_list[j][2], task[1],
                                                      task[2], task[5])

                            utility_sequence.append(utility)

                            task_wait_time += task[2] - executing_task_list[j][2]

                            break

                    # set executing task list i to empty, means core i need to find a new task to process

                    executing_task_list[j] = [0, 0, 0, 0, 0, 0]

        # all tasks in task pool wait 1 unit

        for task in task_pool:
            task[2] -= 1

        for task in task_pool:

            if task[2] < 0:
                error_flag = 1

                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        for task in executing_task_list:

            if task[2] < 0:
                error_flag = 1

                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        if error_flag == 1:
            break

        print('executing task:')

        for item in executing_task_list:
            print(item)

        print('task pool:')

        print(task_pool)

        print('')

        task_pool_length.append(len(task_pool))

        utility_detail.append(sum(utility_sequence))

    print('')

    print('the execute sequence of X is:{}'.format(execute_sequence.__len__()))

    print('')

    print('the accumulate utility is {}'.format(accumulate_utility))

    print('')

    print('the utility of X is:{}'.format(sum(utility_sequence)))

    print('')

    return len(execute_sequence), accumulate_utility, utility_sequence, task_decline_information, error_flag, task_wait_time


def global_DPDA(task_arrive_list, total_timeslot, core_number, times, max_lenth_of_task_pool):
    task_pool = np.empty((0, 6))

    execute_sequence = []

    utility_sequence = []

    task_wait_time = 0

    task_arrive_list_sorted = sort_by_id(task_arrive_list)

    task_decline_number_EDF = 0

    task_decline_information = np.zeros(5)

    task_decline_id = []

    accumulate_utility = 0

    utility_detail = []

    task_pool_length = []

    executing_task = [0, 0, 0, 0, 0, 0]

    executing_task_list = []

    for i in range(core_number):
        executing_task_list.append(executing_task)

    error_flag = 0

    # system runs total time slots

    for timeslot in range(total_timeslot):

        print('DPDA, {}th time, time slot {}'.format(times, timeslot))

        if (timeslot + 1) == 100:
            task_decline_information[0] += task_decline_number_EDF

        if (timeslot + 1) == 200:
            task_decline_information[1] += task_decline_number_EDF

        if (timeslot + 1) == 300:
            task_decline_information[2] += task_decline_number_EDF

        if (timeslot + 1) == 400:
            task_decline_information[3] += task_decline_number_EDF

        if (timeslot + 1) == 500:
            task_decline_information[4] += task_decline_number_EDF

        # for each time slot, find if there are tasks coming

        for i in range(len(task_arrive_list_sorted)):

            # if there are tasks coming, try to insert the tasks into task pool

            # global part

            if task_arrive_list_sorted[i][4] == timeslot:

                # if task pool is not full

                if len(task_pool) < max_lenth_of_task_pool:

                    task_pool = np.insert(task_pool, len(task_pool), task_arrive_list_sorted[i], 0)

                    # use admission control to jude if the task if acceptable

                    if global_admission_control(task_pool, core_number, executing_task_list) == 0:

                        accumulate_utility += task_arrive_list_sorted[i][1] * task_arrive_list_sorted[i][5]

                    # task i is not acceptable for the current task pool

                    else:

                        task_decline_number_EDF += 1

                        task_decline_id.append(task_arrive_list_sorted[i][0])

                        # delete task i

                        for j in range(len(task_pool)):

                            if task_pool[j][0] == task_arrive_list_sorted[i][0]:
                                task_pool = np.delete(task_pool, j, 0)

                                break

                else:

                    task_decline_number_EDF += 1

                    task_decline_id.append(task_arrive_list_sorted[i][0])

            if task_arrive_list_sorted[i][4] > timeslot:
                break

            # global part over

        # update all the cores, if a core is empty

        task_pool = sort_by_average_utility(task_pool, task_arrive_list_sorted)

        for i in range(core_number):

            if executing_task_list[i][1] == 0:

                if len(task_pool) > 0:

                    for j in range(len(task_pool)):

                        test_pool = copy.deepcopy(task_pool)

                        executing_task_list[i] = test_pool[j]

                        test_pool = np.delete(test_pool, j, 0)

                        if global_admission_control(test_pool, core_number, executing_task_list) == 0:

                            task_pool = np.delete(task_pool, j, 0)

                            break

                        else:

                            executing_task_list[i] = [0, 0, 0, 0, 0, 0]

        # all task pools and cores are updated, process the tasks now

        for j in range(core_number):

            # there is a task executing in core i

            if executing_task_list[j][1] > 0:

                # task in the core i is processed for 1 unit

                executing_task_list[j][1] -= 1

                # if the executing task is finished

                if executing_task_list[j][1] == 0:
                    # add the task into executed sequence

                    execute_sequence.append(executing_task_list[j][0])

                    # compute the utility of executing task

                    for task in task_arrive_list:

                        if task[0] == executing_task_list[j][0]:
                            utility = compute_utility(task[3], task[2] - executing_task_list[j][2], task[1],
                                                      task[2], task[5])

                            utility_sequence.append(utility)

                            task_wait_time += task[2] - executing_task_list[j][2]

                            break

                    # set executing task list i to empty, means core i need to find a new task to process

                    executing_task_list[j] = [0, 0, 0, 0, 0, 0]

        # all tasks in task pool wait 1 unit

        for task in task_pool:
            task[2] -= 1

        for task in task_pool:

            if task[2] < 0:
                error_flag = 1
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        for task in executing_task_list:

            if task[2] < 0:
                error_flag = 1
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        if error_flag == 1:
            break

        print('executing task:')

        for item in executing_task_list:
            print(item)

        print('task pool:')

        print(task_pool)

        print('')

        task_pool_length.append(len(task_pool))

        utility_detail.append(sum(utility_sequence))

    print('')

    print('the execute sequence of PLA is:{}'.format(execute_sequence.__len__()))

    print('')

    print('the accumulate utility is {}'.format(accumulate_utility))

    print('')

    print('the utility of PLA is:{}'.format(sum(utility_sequence)))

    print('')

    return len(execute_sequence), accumulate_utility, utility_sequence, task_decline_information, error_flag, task_wait_time, utility_detail, task_pool_length


def global_EDF(task_arrive_list, total_timeslot, core_number, times, max_lenth_of_task_pool):
    task_pool = np.empty((0, 6))

    execute_sequence = []

    utility_sequence = []

    task_wait_time = 0

    task_arrive_list_sorted = sort_by_id(task_arrive_list)

    task_decline_number_EDF = 0

    task_decline_information = np.zeros(5)

    task_decline_id = []

    accumulate_utility = 0

    utility_detail = []

    task_pool_length = []

    executing_task = [0, 0, 0, 0, 0, 0]

    executing_task_list = []

    for i in range(core_number):
        executing_task_list.append(executing_task)

    error_flag = 0



    # system runs total time slots

    for timeslot in range(total_timeslot):

        gc.collect()

        print('EDF, {}th time, time slot {}'.format(times, timeslot))

        if (timeslot + 1) == 100:
            task_decline_information[0] += task_decline_number_EDF

        if (timeslot + 1) == 200:
            task_decline_information[1] += task_decline_number_EDF

        if (timeslot + 1) == 300:
            task_decline_information[2] += task_decline_number_EDF

        if (timeslot + 1) == 400:
            task_decline_information[3] += task_decline_number_EDF

        if (timeslot + 1) == 500:
            task_decline_information[4] += task_decline_number_EDF

        # for each time slot, find if there are tasks coming

        for i in range(len(task_arrive_list_sorted)):

            # if there are tasks coming, try to insert the tasks into task pool

            # global part

            if task_arrive_list_sorted[i][4] == timeslot:

                # if task pool is not full

                if len(task_pool) < max_lenth_of_task_pool:

                    task_pool = np.insert(task_pool, len(task_pool), task_arrive_list_sorted[i], 0)

                    # use admission control to jude if the task if acceptable

                    if global_admission_control(task_pool, core_number, executing_task_list) == 0:

                        accumulate_utility += task_arrive_list_sorted[i][1] * task_arrive_list_sorted[i][5]

                    # task i is not acceptable for the current task pool

                    else:

                        task_decline_number_EDF += 1

                        task_decline_id.append(task_arrive_list_sorted[i][0])

                        # delete task i

                        for j in range(len(task_pool)):

                            if task_pool[j][0] == task_arrive_list_sorted[i][0]:
                                task_pool = np.delete(task_pool, j, 0)

                                break

                else:

                    task_decline_number_EDF += 1

                    task_decline_id.append(task_arrive_list_sorted[i][0])

            if task_arrive_list_sorted[i][4] > timeslot:
                break

            # global part over

        print('executing task:')

        if timeslot == 1:
            executing_task_list = np.reshape(executing_task_list, (5, 6))

        print(executing_task_list)

        print('task pool:')

        print(task_pool)

        # update all the cores, if a core is empty

        for i in range(core_number):

            if executing_task_list[i][1] == 0:

                if len(task_pool) > 0:
                    task_pool = sort_by_deadline(task_pool)

                    executing_task_list[i] = task_pool[0]

                    task_pool = np.delete(task_pool, 0, 0)

        # all task pools and cores are updated, process the tasks now

        for j in range(core_number):

            # there is a task executing in core i

            if executing_task_list[j][1] > 0:

                # task in the core i is processed for 1 unit

                executing_task_list[j][1] -= 1

                # if the executing task is finished

                if executing_task_list[j][1] == 0:
                    # add the task into executed sequence

                    execute_sequence.append(executing_task_list[j][0])

                    # compute the utility of executing task

                    for task in task_arrive_list:

                        if task[0] == executing_task_list[j][0]:
                            utility = compute_utility(task[3], task[2] - executing_task_list[j][2], task[1],
                                                      task[2], task[5])

                            utility_sequence.append(utility)

                            task_wait_time += task[2] - executing_task_list[j][2]

                            break

                    # set executing task list i to empty, means core i need to find a new task to process

                    executing_task_list[j] = [0, 0, 0, 0, 0, 0]

        # all tasks in task pool wait 1 unit

        for task in task_pool:
            task[2] -= 1

        for task in task_pool:

            if task[2] < 0:
                error_flag = 1
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        for task in executing_task_list:

            if task[2] < 0:
                error_flag = 1
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        if error_flag == 1:
            break

        print('')

        task_pool_length.append(len(task_pool))

        utility_detail.append(sum(utility_sequence))

    print('')

    print('the execute sequence of PLA is:{}'.format(execute_sequence.__len__()))

    print('')

    print('the accumulate utility is {}'.format(accumulate_utility))

    print('')

    print('the utility of PLA is:{}'.format(sum(utility_sequence)))

    print('')

    return len(execute_sequence), accumulate_utility, utility_sequence, task_decline_information, error_flag, task_wait_time, utility_detail, task_pool_length


def RL_qtable(task_arrive_list, total_timeslot, core_number, times, max_lenth_of_task_pool):
    task_pool = np.empty((0, 6))

    execute_sequence = []

    utility_sequence = []

    task_arrive_list_sorted = sort_by_id(task_arrive_list)

    task_decline_number_EDF = 0

    task_decline_information = np.zeros(5)

    task_decline_id = []

    accumulate_utility = 0

    utility_detail = []

    task_pool_length = []

    executing_task = [0, 0, 0, 0, 0, 0]

    executing_task_list = []

    for i in range(core_number):
        executing_task_list.append(executing_task)

    error_flag = 0

    # system runs total time slots

    for timeslot in range(total_timeslot):

        print('EDF, {}th time, time slot {}'.format(times, timeslot))

        if (timeslot + 1) == 100:
            task_decline_information[0] += task_decline_number_EDF

        if (timeslot + 1) == 200:
            task_decline_information[1] += task_decline_number_EDF

        if (timeslot + 1) == 300:
            task_decline_information[2] += task_decline_number_EDF

        if (timeslot + 1) == 400:
            task_decline_information[3] += task_decline_number_EDF

        if (timeslot + 1) == 500:
            task_decline_information[4] += task_decline_number_EDF

        # for each time slot, find if there are tasks coming

        for i in range(len(task_arrive_list_sorted)):

            # if there are tasks coming, try to insert the tasks into task pool

            # global part

            if task_arrive_list_sorted[i][4] == timeslot:

                # if task pool is not full

                if len(task_pool) < max_lenth_of_task_pool:

                    task_pool = np.insert(task_pool, len(task_pool), task_arrive_list_sorted[i], 0)

                    # use admission control to jude if the task if acceptable

                    if global_admission_control(task_pool, core_number, executing_task_list) == 0:

                        accumulate_utility += task_arrive_list_sorted[i][1] * task_arrive_list_sorted[i][5]

                    # task i is not acceptable for the current task pool

                    else:

                        task_decline_number_EDF += 1

                        task_decline_id.append(task_arrive_list_sorted[i][0])

                        # delete task i

                        for j in range(len(task_pool)):

                            if task_pool[j][0] == task_arrive_list_sorted[i][0]:
                                task_pool = np.delete(task_pool, j, 0)

                                break

                else:

                    task_decline_number_EDF += 1

                    task_decline_id.append(task_arrive_list_sorted[i][0])

            if task_arrive_list_sorted[i][4] > timeslot:
                break

            # global part over

        # update all the cores, if a core is empty

        for i in range(core_number):

            if executing_task_list[i][1] == 0:

                if len(task_pool) > 0:
                    q_table = rl(task_pool, task_arrive_list_sorted)

                    executing_task_list[i] = task_pool[0]

                    task_pool = np.delete(task_pool, 0, 0)

        # all task pools and cores are updated, process the tasks now

        for j in range(core_number):

            # there is a task executing in core i

            if executing_task_list[j][1] > 0:

                # task in the core i is processed for 1 unit

                executing_task_list[j][1] -= 1

                # if the executing task is finished

                if executing_task_list[j][1] == 0:
                    # add the task into executed sequence

                    execute_sequence.append(executing_task_list[j][0])

                    # compute the utility of executing task

                    for task in task_arrive_list:

                        if task[0] == executing_task_list[j][0]:
                            utility = compute_utility(task[3], task[2] - executing_task_list[j][2], task[1],
                                                      task[2], task[5])

                            utility_sequence.append(utility)

                            break

                    # set executing task list i to empty, means core i need to find a new task to process

                    executing_task_list[j] = [0, 0, 0, 0, 0, 0]

        # all tasks in task pool wait 1 unit

        for task in task_pool:
            task[2] -= 1

        for task in task_pool:

            if task[2] < 0:
                error_flag = 1
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        for task in executing_task_list:

            if task[2] < 0:
                error_flag = 1
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        if error_flag == 1:
            break

        print('this is time slot {}'.format(timeslot))

        print('executing task:')

        for item in executing_task_list:
            print(item)

        print('task pool:')

        print(task_pool)

        print('')

        task_pool_length.append(len(task_pool))

        utility_detail.append(sum(utility_sequence))

    print('')

    print('the execute sequence of RL is:{}'.format(execute_sequence.__len__()))

    print('')

    print('the accumulate utility is {}'.format(accumulate_utility))

    print('')

    print('the utility of RL is:{}'.format(sum(utility_sequence)))

    print('')

    return len(execute_sequence), accumulate_utility, sum(utility_sequence), task_decline_information, error_flag


def count_rest_time(task_pool, core_number):
    task_pool_test = copy.deepcopy(task_pool)

    task_pool_test = sort_by_computation(task_pool_test)

    total_time = 0

    while len(task_pool_test) > 0:

        temparary_delete_list = []

        if len(task_pool_test) <= core_number:

            for i in range(len(task_pool_test)):

                task_pool_test[i][1] -= 1

                if task_pool_test[i][1] == 0:
                    temparary_delete_list.append(i)

        if len(task_pool_test) > core_number:

            for i in range(core_number):

                task_pool_test[i][1] -= 1

                if task_pool_test[i][1] == 0:
                    temparary_delete_list.append(i)

        task_pool_test = np.delete(task_pool_test, temparary_delete_list, 0)

        total_time += 1

    return total_time


def count_rest_time_one_task_pool(task_pool, task_id):
    task_pool_test = copy.deepcopy(task_pool)

    accumulate_time = 0

    for i in range(len(task_pool_test)):

        if task_pool_test[i][0] == task_id:
            task_pool_test = np.delete(task_pool_test, i, 0)

            break

    task_pool_test = sort_by_computation(task_pool_test)

    for i in range(len(task_pool_test)):
        accumulate_time += task_pool_test[i][1] * (len(task_pool_test) - 1)

    return accumulate_time


def t1(t):
    times = t

    print('the {}th second'.format(times))


def t2():
    t = 0
    while 1:
        t1(t)
        time.sleep(1)
        t += 1

# np.random.seed(1)
mu = 0
sigma = 0.06

def main():
    # t = threading.Thread(target=t2)
    # t.start()

    task_arrival_expectation = 0.225
    computation_expectation = 20
    slack_expectation = 30
    core_number = 5
    utility_type = 1
    max_lenth_of_mutiple_task_pool = 10
    N = 1
    total_timeslot = 50
    task_pool = np.empty((0, 6))
    max_lenth_of_one_task_pool = 5
    executed_task_number_sequence_GEDF = []
    executed_task_number_sequence_GDPDA = []
    executed_task_number_sequence_GPLA = []
    executed_task_number_sequence_delta = []

    accumulate_utility_sequence_GEDF = []
    accumulate_utility_sequence_GDPDA = []
    accumulate_utility_sequence_GPLA = []
    accumulate_utility_sequence_delta = []

    total_utility_sequence_GEDF = []
    total_utility_sequence_GDPDA = []
    total_utility_sequence_GPLA = []
    total_utility_sequence_delta = []

    task_decline_number_GEDF = []
    task_decline_number_GDPDA = []
    task_decline_number_GPLA = []
    task_decline_number_delta = []

    task_wait_time_GEDF = []
    task_wait_time_GDPDA = []
    task_wait_time_GPLA = []
    task_wait_time_delta = []
    total_number_of_task = 0

    # np.random.seed(2)

    # run system N times



    for times in range(N):

        X_axix = []

        task_id = 0

        """
        creates the information of tasks
        """

        task_arrive_list = np.empty((0, 6))

        for timeslot in range(total_timeslot):

            X_axix.append(timeslot)

            # check if there are tasks need to update their slack

            if len(task_pool) > 0:

                for i in range(len(task_pool)):

                    # the probability for changing slack is 5%

                    if np.random.randint(100) <= 4:

                        for item in task_arrive_list:

                            if item[0] == task_pool[i][0]:
                                item[2] += 10

            task_arrival_number = np.random.poisson(task_arrival_expectation)

            for i in range(task_arrival_number):
                task_i = creat_task(task_id, computation_expectation, slack_expectation, timeslot, utility_type)

                task_arrive_list = np.insert(task_arrive_list, len(task_arrive_list), task_i, 0)

                task_id += 1

        total_number_of_task += len(task_arrive_list)

        """
        DRL
        """
        executed_task_number, accumulate_utility, utility, task_decline_number, error_flag, task_wait_time, utility_detail_DRL, task_pool_length_DRL = global_DRL(
            task_arrive_list, total_timeslot, core_number, times, max_lenth_of_mutiple_task_pool)

        executed_task_number_sequence_delta.append(executed_task_number)

        accumulate_utility_sequence_delta.append(accumulate_utility)

        total_utility_sequence_delta.append(sum(utility))

        task_decline_number_delta.append(task_decline_number)

        task_wait_time_delta.append(task_wait_time / executed_task_number)

        if error_flag == 1:
            break


        """
        EDF_global
        """
        executed_task_number, accumulate_utility, utility, task_decline_number, error_flag, task_wait_time, utility_detail_EDF, task_pool_length_EDF = global_EDF(
            task_arrive_list, total_timeslot, core_number, times, max_lenth_of_mutiple_task_pool)

        executed_task_number_sequence_GEDF.append(executed_task_number)

        accumulate_utility_sequence_GEDF.append(accumulate_utility)

        total_utility_sequence_GEDF.append(sum(utility))

        task_decline_number_GEDF.append(task_decline_number)

        task_wait_time_GEDF.append(task_wait_time / executed_task_number)

        if error_flag == 1:
            break



        """
        DPDA_global
        """
        executed_task_number, accumulate_utility, utility, task_decline_number, error_flag, task_wait_time, utility_detail_DPDA, task_pool_length_DPDA = global_DPDA(
            task_arrive_list, total_timeslot, core_number, times, max_lenth_of_mutiple_task_pool)

        executed_task_number_sequence_GDPDA.append(executed_task_number)

        accumulate_utility_sequence_GDPDA.append(accumulate_utility)

        total_utility_sequence_GDPDA.append(sum(utility))

        task_decline_number_GDPDA.append(task_decline_number)

        task_wait_time_GDPDA.append(task_wait_time / executed_task_number)

        if error_flag == 1:
            break

        """
        PLA_global
        """
        executed_task_number, accumulate_utility, utility, task_decline_number, error_flag, task_wait_time, utility_detail_PLA, task_pool_length_PLA = global_PLA(task_arrive_list, total_timeslot, core_number, times, max_lenth_of_mutiple_task_pool)

        executed_task_number_sequence_GPLA.append(executed_task_number)

        accumulate_utility_sequence_GPLA.append(accumulate_utility)

        total_utility_sequence_GPLA.append(sum(utility))

        task_decline_number_GPLA.append(task_decline_number)

        task_wait_time_GPLA.append(task_wait_time / executed_task_number)

        if error_flag == 1:
            break


    if error_flag == 0:

        print('results of GEDF')

        print('the average executed task number is {}'.format(
            sum(executed_task_number_sequence_GEDF) / len(executed_task_number_sequence_GEDF)))

        print('the average declined task number is {}'.format(
            sum(task_decline_number_GEDF) / len(task_decline_number_GEDF)))

        print('the average accumulated utility is {}'.format(
            sum(accumulate_utility_sequence_GEDF) / len(accumulate_utility_sequence_GEDF)))

        print('the average total utility is {}'.format(
            sum(total_utility_sequence_GEDF) / len(total_utility_sequence_GEDF)))

        print('the average utility ratio is {}'.format(
            (sum(total_utility_sequence_GEDF) / len(total_utility_sequence_GEDF)) / (
                        sum(accumulate_utility_sequence_GEDF) / len(accumulate_utility_sequence_GEDF))))

        print('the average task wait time is {}'.format(sum(task_wait_time_GEDF) / len(task_wait_time_GEDF)))

        print('')

        print('results of GDPDA')

        print('the average executed task number is {}'.format(
            sum(executed_task_number_sequence_GDPDA) / len(executed_task_number_sequence_GDPDA)))

        print('the average declined task number is {}'.format(
            sum(task_decline_number_GDPDA) / len(task_decline_number_GDPDA)))

        print('the average accumulated utility is {}'.format(
            sum(accumulate_utility_sequence_GDPDA) / len(accumulate_utility_sequence_GDPDA)))

        print('the average total utility is {}'.format(
            sum(total_utility_sequence_GDPDA) / len(total_utility_sequence_GDPDA)))

        print('the average utility ratio is {}'.format(
            (sum(total_utility_sequence_GDPDA) / len(total_utility_sequence_GDPDA)) / (
                        sum(accumulate_utility_sequence_GDPDA) / len(accumulate_utility_sequence_GDPDA))))

        print('the average task wait time is {}'.format(sum(task_wait_time_GDPDA) / len(task_wait_time_GDPDA)))

        print('')

        print('results of GPLA')

        print('the average executed task number is {}'.format(sum(executed_task_number_sequence_GPLA) / len(executed_task_number_sequence_GPLA)))

        print('the average declined task number is {}'.format(sum(task_decline_number_GPLA) / len(task_decline_number_GPLA)))

        print('the average accumulated utility is {}'.format(sum(accumulate_utility_sequence_GPLA) / len(accumulate_utility_sequence_GPLA)))

        print('the average total utility is {}'.format(sum(total_utility_sequence_GPLA) / len(total_utility_sequence_GPLA)))

        print('the average utility ratio is {}'.format((sum(total_utility_sequence_GPLA) / len(total_utility_sequence_GPLA)) / (sum(accumulate_utility_sequence_GPLA) / len(accumulate_utility_sequence_GPLA))))

        print('the average task wait time is {}'.format(sum(task_wait_time_GPLA) / len(task_wait_time_GPLA)))

        print('')

        print('results of DRL')

        print('the average executed task number is {}'.format(
            sum(executed_task_number_sequence_delta) / len(executed_task_number_sequence_delta)))

        print('the average declined task number is {}'.format(
            sum(task_decline_number_delta) / len(task_decline_number_delta)))

        print('the average accumulated utility is {}'.format(
            sum(accumulate_utility_sequence_delta) / len(accumulate_utility_sequence_delta)))

        print('the average total utility is {}'.format(
            sum(total_utility_sequence_delta) / len(total_utility_sequence_delta)))

        print('the average utility ratio is {}'.format(
            (sum(total_utility_sequence_delta) / len(total_utility_sequence_delta)) / (
                        sum(accumulate_utility_sequence_delta) / len(accumulate_utility_sequence_delta))))

        print('the average task wait time is {}'.format(sum(task_wait_time_delta) / len(task_wait_time_delta)))


    else:

        print('there is an error!!!!!!!!!!!!!!!!!!!')



    # plt.plot(X_axix, utility_detail_G_EDF, color='green', label='utility_detail_G_EDF')
    # plt.plot(X_axix, utility_detail_LB_EDF, color='red', label='utility_detail_LB_EDF')
    # plt.plot(X_axix, task_pool_length_LB_EDF, color='green', label='task_pool_length_LB_EDF')


    X1_axix = copy.deepcopy(X_axix)
    X2_axix = copy.deepcopy(X_axix)
    plt.plot(X1_axix, utility_detail_DRL, color='blue', label='DRL')
    plt.plot(X1_axix, utility_detail_EDF, color='black', label='EDF')
    plt.plot(X1_axix, utility_detail_DPDA, color='yellow', label='DPDA')
    plt.plot(X1_axix, utility_detail_PLA, color='red', label='PLA')
    plt.plot(X2_axix, task_pool_length_DRL, color='blue', label='task pool length of DRL')
    plt.plot(X2_axix, task_pool_length_EDF, color='black', label='task pool length of EDF')
    plt.plot(X2_axix, task_pool_length_DPDA, color='yellow', label='task pool length of DPDA')
    plt.plot(X2_axix, task_pool_length_PLA, color='red', label='task pool length of PLA')


    plt.legend()
    plt.show()



    # X_axix_2 = X_axix

    # plt.plot(X_axix_2, task_pool_length_G_EDF, color='green', label='task_pool_length_G_EDF')
    # plt.plot(X_axix_2, task_pool_length_LB_EDF, color='red', label='task_pool_length_LB_EDF')

    # t.join()



# DQN class

class DQNAgent(object):
    def __init__(self, _env):
        self.env = _env
        # 经验池
        self.memory = []
        self.gamma = 0.9 # decay rate 奖励衰减

        # 控制训练的随机干涉
        self.epsilon = 1  # 随机干涉阈值 该值会随着训练减少
        self.epsilon_decay = .995  # 每次随机衰减0.005
        self.epsilon_min = 0.1  # 随机值干涉的最小值

        self.learning_rate = 0.0001  # 学习率

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

        self._build_model()

    # 创建模型  输入4种状态，预测0/1两种行为分别带来到 奖励值
    def _build_model(self):
        model = tf.keras.Sequential()
        # 第1层卷积，卷积核大小为2*2，16个，25*4为待训练矩阵的大小
        model.add(tf.keras.layers.Conv2D(16, (2, 2), activation='relu', input_shape=(25, 4, 1)))
        model.add(tf.keras.layers.MaxPooling2D((2, 1)))
        # 第2层卷积，卷积核大小为2*2，32个
        model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu'))
        # 展开成1维向量
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        self.model = model

    # 记录经验 推进memory中
    def save_exp(self, _state, _action, _reward, _next_state, _done):
        # 将各种状态 存进memory中
        self.memory.append((_state, _action, _reward, _next_state, _done))

    # 经验池重放  根据尺寸获取  #  这里是训练数据的地方
    def train_exp(self, batch_size):
        # 这句的目的：确保每批返回的数量不会超出memory实际的存量，防止错误
        batches = min(batch_size, len(self.memory))  # 返回不大于实际memory.len 的数
        # 从len(self.memory）中随机选出batches个数
        batches = np.random.choice(len(self.memory), batches)

        for i in batches:
            # 从经验数组中 取出相对的参数 状态，行为，奖励，即将发生的状态，结束状态
            _state, _action, _reward, _next_state, _done = self.memory[i]

            # 获取当前 奖励
            y_reward = _reward
            # 如果不是结束的状态，取得未来的折扣奖励
            if not _done:
                # _target = 经验池.奖励 + gamma衰退值 * (model根据_next_state预测的结果)[0]中最大（对比所有操作中，最大值）
                # 根据_next_state  预测取得  当前的回报 + 未来的回报 * 折扣因子（gama）
                y_reward = _reward + self.gamma * self.model.predict(_next_state[:,1:5])
                # print('y_action', y_action)  # 1.5389154434204102

            # 获取，根据当前状态推断的 行为 input 15*6,output 385
            _y = self.model.predict(_state[:,1:5])
            # print(_action, y_reward, _y[0][_action], '_y', _y)  # [[0.08838317 0.16991007]]
            # 更新 将 某行为预测的 回报，分配到相应到 行为中  （_action = 1/0）
            _y[0][_action] = y_reward
            # 训练  x： 4 当前状态  _y[0]：2
            self.model.fit(_state[:,1:5], _y, epochs=1, verbose=0)

        # 循环训练，每次调用fit都会创建一个新的图，导致内存爆炸，这里解决相关问题
        K.clear_session()

        # 随着训练的继续，每次被随机值干涉的几率减少 * epsilon_decay倍数(0.001)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # 输入状态，返回应该采取的动作。随机干涉会随着时间的推进，越来越少。
    def act(self, _state):  # 返回回报最大的奖励值，或 随机数
        # 随机返回0-1的数，
        # 随着训练的增加，渐渐减少随机
        if np.random.rand() <= self.epsilon:
            # print('000000000',self.env.action_space.sample())
            return self.env.action_space.sample()
        else:
            # 使用预测值    返回，回报最大到最大的那个
            act_values = self.model.predict(_state.reshape((1,15,6,1)))
            return np.argmax(act_values[0])  # returns action



if __name__ == '__main__':
    # 为agent初始化gym环境参数
    env = gym.make('taskschduling-v0')

    # 游戏结束规则：杆子角度为±12， car移动距离为±2.4，分数限制为最高200分
    agent = DQNAgent(env)
    # 保存模型到脚本所在目录，如果已有模型就加载，继续训练
    folder = os.getcwd()
    imageList = os.listdir(folder)
    for item in imageList:
        if os.path.isfile(os.path.join(folder, item)):
            if item == 'scheduling.h5':
                agent.model = tf.keras.models.load_model('scheduling.h5')

    main()










