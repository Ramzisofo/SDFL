import sys
import numpy as np
from score import simplify_eq
import random
from itertools import permutations
import re

from collections import defaultdict


class SdflBase:

    def __init__(self, base_grammars, aug_grammars, nt_nodes, max_len, max_module, aug_grammars_allowed,
                 func_score, graph=None, exploration_rate=1 / np.sqrt(2), eta=0.999, dim=3):

        self.base_grammars = base_grammars
        self.grammars = base_grammars + [x for x in aug_grammars if x not in base_grammars]
        self.nt_nodes = nt_nodes
        self.max_len = max_len
        self.max_module = max_module
        self.max_aug = aug_grammars_allowed
        self.good_modules = []
        self.score = func_score
        self.ch_action = 0
        self.dim = dim
        self.graph = graph # dictionary {node: list of (connected node, corresponding coefficient)}
        self.exploration_rate = exploration_rate
        self.UCBs = defaultdict(lambda: np.zeros(len(self.grammars)))
        self.QN = defaultdict(lambda: np.zeros(2))
        self.scale = 0
        self.eta = eta

    def valid_prods(self, Node, state=None):
        """
        Get index of all possible production rules starting with a given node
        """
        valid_grams = [self.grammars.index(x) for x in self.grammars if x.startswith(Node)]

        if state is not None:

                if '/A' in state[-8:]:
                    valid_grams.pop(self.grammars.index('A->A/A'))
                    grams = self.grammars.copy()
                    grams.pop(self.grammars.index('A->A/A'))

        return valid_grams

    def tree_to_eq(self, prods):
        """
        Convert a parse tree to equation form
        """
        seq = ['f']
        for prod in prods:
            if str(prod[0]) == 'Nothing':
                break
            for ix, s in enumerate(seq):
                if s == prod[0]:
                    seq = seq[:ix] + list(prod[3:]) + seq[ix + 1:]
                    break
        try:
            return ''.join(seq)
        except:
            return ''

    def state_to_seq(self, state):
        """
        Convert the state to sequence of index
        """
        aug_grammars = ['f->A'] + self.grammars
        seq = np.zeros(self.max_len)
        prods = state.split(',')
        for i, prod in enumerate(prods):
            seq[i] = aug_grammars.index(prod)
        return seq

    def state_to_onehot(self, state):
        """
        Convert the state to one hot matrix
        """
        aug_grammars = ['f->A'] + self.grammars
        state_oh = np.zeros([self.max_len, len(aug_grammars)])
        prods = state.split(',')
        for i in range(len(prods)):
            state_oh[i, aug_grammars.index(prods[i])] = 1

        return state_oh

    def get_ntn(self, prod, prod_idx):
        """
        Get all the non-terminal nodes from right-hand side of a production rule grammar
        """
        if prod_idx >= len(self.base_grammars):
            return []
        else:
            return [i for i in prod[3:] if i in self.nt_nodes]

    def get_unvisited(self, state, node):
        """
        Get index of all unvisited child
        """
        valid_action = self.valid_prods(node, state)
        return [a for a in valid_action if self.QN[state + ',' + self.grammars[a]][1] == 0]

    def print_solution(self, solu, i_episode):
        print('Episode', i_episode, solu)

    def step(self, state, action_idx, ntn, pr=False):
        """
        state: all production rules
        action_idx: index of grammar starts from the current Non-terminal Node
        tree: the current tree
        ntn: all remaining non-terminal nodes


        This defines one step of Parse Tree traversal
        return tree (next state), remaining non-terminal nodes, reward, and if it is done
        """
        action = self.grammars[action_idx]
        state = state + ',' + action
        ntn = self.get_ntn(action, action_idx) + ntn[1:]

        if not ntn:

            # eq_comp = simplify_eq(self.tree_to_eq(state.split(',')))
            eq = self.tree_to_eq(state.split(','))

            count_nondim = np.array([eq.count('x'+ str(dim_p)) for dim_p in range(1, self.dim+1) if dim_p != 1])

            org_eq = eq

            if self.graph is None:
                if np.any(count_nondim != 0): 
    
                    for i in range(3, self.dim+1):
                        # org_eq = simplify_eq(org_eq)
                        eq_temp = re.sub(r'x' + str(2)+' ', 'x' + str(i) + ' ', org_eq)
                        eq += '+' + eq_temp
    
                
    
                eq_vec = ['(' + eq  + ' )' for _ in range(self.dim)]
                for dim in range(1, self.dim):
                    modified_eq = re.sub(r'x' + str(1) + ' ', 'TEMP', eq_vec[dim])  # Replace '1' with temporary marker
                    modified_eq = modified_eq.replace('x' + str(dim+1) + ' ', 'x' + '1 ')  # Replace str(i) with '1'
                    modified_eq = modified_eq.replace('TEMP', 'x' + str(dim+1) + ' ')  # Replace temporary marker with str(i)
                    eq_vec[dim] = modified_eq
                    
            else:

                # Enforcing permutation invariance 

                eq_vec = [eq for _ in range(self.dim)]

                if np.any(count_nondim != 0):

                    eq = ' '
                    dim = 1
                    for i, coef in self.graph[dim]:
                        # org_eq = simplify_eq(org_eq)
                        if i == dim and 'x1' not in eq:
                            continue
                        eq_temp = re.sub(r'x' + str(2) + ' ', 'x' + str(i) + ' ', org_eq)
                        eq += '+ ' + str(coef) + '*(' + eq_temp + ' )'

                    eq_vec[dim-1] = eq
                    for dim in range(2, self.dim):
                        org_eq = re.sub(r'x' + str(1) + ' ', 'x' + str(dim+1) + ' ', org_eq)
                        eq = ' '
                        for i, coef in self.graph[dim]:
                            # org_eq = simplify_eq(org_eq)
                            if i == dim and ('x'+str(dim) not in eq):
                                continue
                            eq_temp = re.sub(r'x' + str(2) + ' ', 'x' + str(i) + ' ', org_eq)
                            eq += '+ ' + str(coef) + '*('+  eq_temp + ' )'
                            
                        eq_vec[dim] = eq

                    print("eq_vec = ", eq_vec)
                else:
                    for dim in range(1, self.dim):
                        modified_eq = re.sub(r'x' + str(1) + ' ', 'TEMP',
                                             eq_vec[dim])  
                        modified_eq = modified_eq.replace('x' + str(dim + 1) + ' ',
                                                          'x' + '1 ') 
                        modified_eq = modified_eq.replace('TEMP', 'x' + str(
                            dim + 1) + ' ')  
                        eq_vec[dim] = modified_eq
                    print("eq_vec ELSE = ", eq_vec)

            reward, eq = self.score(eq_vec, eq_vec,
                                    [len(state.split(',')) for dim in range(self.dim)],
                                    self.data_sample, eta=self.eta)
            return state, ntn, reward, True, eq[0]
        else:

            return state, ntn, 0, False, None

    def rollout(self, num_play, state_initial, ntn_initial, pr=False):
        """
        Perform a n-play rollout simulation, get the maximum reward
        """
        best_eq = ''
        best_r = 0
        for n in range(num_play):
            done = False
            state = state_initial
            ntn = ntn_initial

            while not done:
                valid_index = self.valid_prods(ntn[0], state)
                action = np.random.choice(valid_index)
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn, pr)
                state = next_state
                ntn = ntn_next
                if state.count(',') >= self.max_len:
                    break

            if done:
                if reward > best_r:
                    self.update_modules(next_state, reward, eq)
                    best_eq = eq
                    best_r = reward
                    print(" done = ", done, " next state = ",
                      [simplify_eq(self.tree_to_eq(state.split(',')))], " reward = ",
                      reward, " best_r = ", best_r)

        return best_r, best_eq

    def update_ucb_mcts(self, state, action):
        """
        Get the ucb score for a given child of current node
        """
        next_state = state + ',' + action
        Q_child = self.QN[next_state][0]
        N_parent = self.QN[state][1]
        N_child = self.QN[next_state][1]
        return Q_child / N_child + self.exploration_rate * np.sqrt(np.log(N_parent) / N_child)

    def update_QN_scale(self, new_scale):
        """
        Update the Q values self.scaled by the new best reward.
        """

        if self.scale != 0:
            for s in self.QN:
                self.QN[s][0] *= (self.scale / new_scale)

        self.scale = new_scale

    def backpropogate(self, state, action_index, reward):
        """
        Update the Q, N and ucb for all corresponding decedent after a complete rollout
        """

        action = self.grammars[action_index]
        if self.scale != 0:
            self.QN[state + ',' + action][0] += reward / self.scale
        else:
            self.QN[state + ',' + action][0] += 0
        self.QN[state + ',' + action][1] += 1

        while state:
            if self.scale != 0:
                self.QN[state][0] += reward / self.scale
            else:
                self.QN[state][0] += 0
            self.QN[state][1] += 1
            self.UCBs[state][self.grammars.index(action)] = self.update_ucb_mcts(state, action)
            if ',' in state:
                state, action = state.rsplit(',', 1)
            else:
                state = ''

    def get_policy1(self, nA):
        """
        Creates an policy based on ucb score.
        """

        def policy_fn(state, node):
            valid_action = self.valid_prods(node, state)

            # collect ucb scores for all valid actions
            policy_valid = []

            sum_ucb = sum(self.UCBs[state][valid_action])

            for a in valid_action:
                policy_mcts = self.UCBs[state][a] / sum_ucb
                policy_valid.append(policy_mcts)

            # if all ucb scores identical, return uniform policy
            if len(set(policy_valid)) == 1:
                A = np.zeros(nA)
                A[valid_action] = float(1 / len(valid_action))
                return A

            # return action with largest ucb score
            A = np.zeros(nA, dtype=float)
            best_action = valid_action[np.argmax(policy_valid)]
            A[best_action] += 0.6
            A[valid_action] += float(0.4 / len(valid_action))
            return A

        return policy_fn

    def get_policy2(self, nA):
        """
        Creates an random policy to select an unvisited child.
        """

        def policy_fn(UC):
            if len(UC) != len(set(UC)):
                print(UC)
                print(self.grammars)
            A = np.zeros(nA, dtype=float)
            A[UC] += float(1 / len(UC))
            return A

        return policy_fn

    def update_modules(self, state, reward, eq):
        """
        If we pass by a concise solution with high score, we store it as an
        single action for future use.
        """
        module = state[5:]
        if state.count(',') <= self.max_module:
            if not self.good_modules:
                self.good_modules = [(module, reward, eq)]
            elif eq not in [x[2] for x in self.good_modules]:
                if len(self.good_modules) < self.max_aug:
                    self.good_modules = sorted(self.good_modules + [(module, reward, eq)], key=lambda x: x[1])
                else:
                    if reward > self.good_modules[0][1]:
                        self.good_modules = sorted(self.good_modules[1:] + [(module, reward, eq)], key=lambda x: x[1])

    def run(self, num_episodes, num_play=50, print_flag=False, print_freq=100):
        """
        Monte Carlo Tree Search algorithm
        """

        nA = len(self.grammars)
        # search history
        states = []

        # The policy we're following:
        # policy1 for fully expanded node and policy2 for not fully expanded node
        policy1 = self.get_policy1(nA)
        policy2 = self.get_policy2(nA)

        reward_his = []
        best_solution = ('nothing', 0)
        pr = False
        for i_episode in range(1, num_episodes + 1):
            if (i_episode) % print_freq == 0 and print_flag:
                print("\rEpisode {}/{}, current best reward {}.".format(i_episode, num_episodes, best_solution[1]),
                      end="")
                sys.stdout.flush()
            print("episode = ", i_episode)
            if i_episode == 20000:
                pr = True
            state = 'f->A'
            ntn = ['A']
            UC = self.get_unvisited(state, ntn[0])

            ##### check scenario: if parent node fully expanded or not ####

            # scenario 1: if current parent node fully expanded, follow policy1
            while not UC:
                action = np.random.choice(np.arange(nA), p=policy1(state, ntn[0]))
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn)
                if state not in states:
                    states.append(state)

                if not done:
                    state = next_state
                    ntn = ntn_next
                    UC = self.get_unvisited(state, ntn[0])

                    if state.count(',') >= self.max_len:
                        UC = []
                        self.backpropogate(state, action, 0)
                        reward_his.append(best_solution[1])
                        break
                else:
                    UC = []
                    if reward > best_solution[1]:
                        self.update_modules(next_state, reward, eq)
                        self.update_QN_scale(reward)
                        best_solution = (eq, reward)
                        print("best solution ", best_solution)

                    self.backpropogate(state, action, reward)
                    reward_his.append(best_solution[1])
                    break

            # scenario 2: if current parent node not fully expanded, follow policy2
            if UC:
                action = np.random.choice(np.arange(nA), p=policy2(UC))
                next_state, ntn_next, reward, done, eq = self.step(state, action, ntn)
                if not done:
                    reward, eq = self.rollout(num_play, next_state, ntn_next, pr)
                    print(" roll out -> out ")
                    if state not in states:
                        states.append(state)

                if reward > best_solution[1]:
                    self.update_QN_scale(reward)
                    best_solution = (eq, reward)

                self.backpropogate(state, action, reward)
                reward_his.append(best_solution[1])

        print("best solution ", best_solution)
        return reward_his, best_solution, self.good_modules