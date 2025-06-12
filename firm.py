from sklearn.linear_model import Ridge
import numpy as np
from utils import *

'''
code for firms
'''
import logging
import os
from datetime import datetime

# Create a directory for logs if it doesn't exist
# log_dir = 'result_dir/log'
# os.makedirs(log_dir, exist_ok=True)

# # Generate a log file name with a timestamp
# log_filename = os.path.join(log_dir, f'tmp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# # Set up logging
# logging.basicConfig(
#     filename=log_filename,  # Dynamic log file name
#     level=logging.DEBUG,  # Logging level
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format of the log messages
#     datefmt='%Y-%m-%d %H:%M:%S'  # Date format
# )


class Firms:
    def __init__(self):
        self.firm_feature_template = load_feature_distributions("firm.json")
        self.frim_features = generate_all_feature_combinations(self.firm_feature_template)
        self.firms_num = len(self.firm_features)
        self.firm_feature_length = len(self.firm_features[0])
        self.candidate_feature_template = load_feature_distributions("candidate.json")
        self.candidate_features = generate_all_feature_combinations(self.candidate_feature_template)
        self.candidate_features_length = len(self.candidate_features[0])
        
    def generate_firms(self):
        firm_list = []
        i = 0
        for feature in self.firm_features:
            feature_vector = encode_one_hot(feature, self.candidate_feature_template)
            new_firm = Firm(i, feature_vector, self.candidate_features, len(feature_vector))
            firm_list.append(new_firm)
            i = i + 1

        return firm_list
        
class Firm:
    def __init__(self, firm_id, firm_features, firm_features_length,candidate_features_length):
        self.firm_id = firm_id
        self.firm_features = firm_features
        self.candidate_feature_length = candidate_features_length
        self.firm_features_length = firm_features_length
        self.epsilon = 0.1
        self.alpha = 0.1
        self.estimate_mean = np.zeros((firm_features_length,candidate_features_length))
        self.estimate_variance = np.ones((firm_features_length,candidate_features_length)) # every arm‘s b
        self.selected_nums = np.zeros((firm_features_length,candidate_features_length))
        self.current_candidate_feature_vector = None
        self.current_candidate_group = None
        self.selected_candidates_feature = []
    

    def select_arm(self):
        # print(self.firm_id, " current_candidate_group", self.current_candidate_group)
        values = []
        for candidate_feature in self.current_candidate_feature_vector:
            interaction = candidate_feature * self.firm_features.T
            sampled_means = np.random.normal(0, 1, (self.firm_features_length, self.candidate_feature_length)) + self.estimate_mean
            # print(sampled_means)
            predicted_reward = np.sum(sampled_means * interaction)
            values.append(predicted_reward)
            # print(values)
        return self.current_candidate_group[np.argmax(values)]

    def update(self, arm_features, reward):
        self.selected_candidates_feature.append(arm_features)
        interaction = arm_features * self.firm_features.T
        
        for i in range(self.firm_features_length):
            if self.firm_features[0][i] == 1:
                for j in range(self.candidate_feature_length):
                   # update the mean
                    if arm_features[0][j] == 1: 
                        update_value = reward[i,j] / interaction[i,j]
                        # print("update_value", update_value)
                        self.estimate_mean[i, j] = (self.estimate_mean[i, j] * self.selected_nums[i,j] + update_value) / (self.selected_nums[i,j] + 1)
                        self.selected_nums[i,j] += 1
                        # not update the variance


def random_argmax(list):
    return np.random.choice(np.flatnonzero(list == list.max()))

class UCBFirmMix:
    def __init__(self, firm_id, firm_features, firm_features_length,candidate_feature_length, hired_num):
        self.firm_id = firm_id
        self.firm_features = firm_features
        self.candidate_feature_length = candidate_feature_length
        self.firm_features_length = firm_features_length
        self.alpha = 1 + np.sqrt(np.log(2/1) / 2)
        self.lambda_reg = 1.0
        self.A = self.lambda_reg * np.identity(self.candidate_feature_length * self.firm_features_length)
        self.b = np.zeros(self.candidate_feature_length * self.firm_features_length)
        self.theta_matrix= np.zeros((firm_features_length,candidate_feature_length), dtype=float)
        self.theta = self.theta_matrix.flatten()
        self.selected_nums = np.zeros((firm_features_length,candidate_feature_length))
        self.current_candidate_feature_vector = None
        self.current_candidate_group = None
        self.selected_candidates_feature = []
        self.hired_num = hired_num
        self.total_num = 1
    
    def select_arm(self):
        def is_index_max_in_list(lst, index):
            # Ensure the index is within the valid range
            if index < 0 or index >= len(lst):
                raise IndexError("Index is out of range")

            # Check if the value at the given index is the maximum in the list
            is_max = lst[index] == max(lst)

            # Return the result
            return is_max
        
        self.total_num += 1
        new_features = [(feat * self.firm_features.T).reshape(-1) for feat in self.current_candidate_feature_vector]
        UCB_values = []
        UCB_mean = []
        UCB_std = []
        i = 0
        for feat in new_features:
            mean = feat.dot(self.theta)
            UCB_mean.append(mean)
            delta = self.alpha * np.sqrt(feat.dot(np.linalg.inv(self.A)).dot(feat) )#* 2 * np.log(self.total_num))
            UCB_std.append(delta)
            UCB_values.append(mean + delta)
            i += 1
        top_indices = [random_argmax(np.array(UCB_values))]
        select_list = [self.current_candidate_group[index] for index in top_indices]
        return select_list

    def select_arm_decentralized(self, theta, A): 
        def is_index_max_in_list(lst, index):
            # Ensure the index is within the valid range
            if index < 0 or index >= len(lst):
                raise IndexError("Index is out of range")

            # Check if the value at the given index is the maximum in the list
            is_max = lst[index] == max(lst)

            # Return the result
            return is_max


        self.total_num += 1
        new_features = [(feat * self.firm_features.T).reshape(-1) for feat in self.current_candidate_feature_vector]
        UCB_values = []
        UCB_mean = []
        UCB_std = []
        i=0
        for feat in new_features:
            mean = feat.dot(theta)
            UCB_mean.append(mean)
            delta = self.alpha * np.sqrt(feat.dot(np.linalg.inv(A)).dot(feat)) #* 2 * np.log(self.total_num * 60))
            UCB_std.append(delta)
            UCB_values.append(mean + delta)
            # logging.debug(f"[decentralized-firm] fetature {self.current_candidate_feature_vector[i]}, mean: {mean}, delta: {delta}, UCB:{mean+delta}")
            i +=1
        # UCB_values = [feat.dot(theta) + self.alpha * np.sqrt(feat.dot(np.linalg.inv(A)).dot(feat)) for feat in new_features]
        top_indices = [random_argmax(np.array(UCB_values))]
        select_list = [self.current_candidate_group[index] for index in top_indices]
        return select_list
    

    def update(self,candidate_id, arm_features, reward):
        # Ridge Regression estimate theta
        # (A + λI)^(-1) * b
        features = (arm_features * self.firm_features.T).reshape(-1)
        self.A += np.outer(features, features)
        self.b += reward * features
        self.theta = np.linalg.inv(self.A).dot(self.b)
        self.selected_candidates_feature.append(arm_features)
        # print(self.firm_id, self.theta)
        # print("selected_candidates_feature", self.selected_candidates_feature)

    def update_decentralized(self, arm_features):
        self.selected_candidates_feature.append(arm_features)


class ThompsonSamplingFirmMix:
    def __init__(self, firm_id, firm_features, firm_features_length, candidate_feature_length, hired_num):
        self.firm_id = firm_id
        self.firm_features = firm_features
        self.candidate_feature_length = candidate_feature_length
        self.firm_features_length = firm_features_length
        self.lambda_reg = 1.0
        self.A = self.lambda_reg * np.identity(self.candidate_feature_length * self.firm_features_length)
        self.b = np.zeros(self.candidate_feature_length * self.firm_features_length)
        self.theta_matrix = np.zeros((firm_features_length, candidate_feature_length), dtype=float)
        self.theta = self.theta_matrix.flatten()
        self.selected_nums = np.zeros((firm_features_length, candidate_feature_length))
        self.current_candidate_feature_vector = None
        self.current_candidate_group = None
        self.selected_candidates_feature = []
        self.hired_num = hired_num
    
    def select_arm(self):
        # Sample theta from the posterior distribution
        new_features = [(feat * self.firm_features.T).reshape(-1) for feat in self.current_candidate_feature_vector]
        # sampled_theta = np.random.multivariate_normal(self.theta, np.linalg.inv(self.A))
        # TS_values = [feat.dot(sampled_theta) for feat in new_features]
        TS_values = []
        for feat in new_features:
            sampled_theta = np.random.multivariate_normal(self.theta, np.linalg.inv(self.A))
            TS_value = feat.dot(sampled_theta)
            TS_values.append(TS_value)
        top_indices = [random_argmax(np.array(TS_values))]
        select_list = [self.current_candidate_group[index] for index in top_indices]
        return select_list
    
    def select_arm_decentralized(self, theta, A):
        # Sample theta from the posterior distribution
        new_features = [(feat * self.firm_features.T).reshape(-1) for feat in self.current_candidate_feature_vector]
        # sampled_theta = np.random.multivariate_normal(theta, np.linalg.inv(A))
        # TS_values = [feat.dot(sampled_theta) for feat in new_features]

        TS_values = []
        for feat in new_features:
            sampled_theta = np.random.multivariate_normal(theta, np.linalg.inv(A))
            TS_value = feat.dot(sampled_theta)
            TS_values.append(TS_value)
        
        top_indices = [random_argmax(np.array(TS_values))]
        select_list = [self.current_candidate_group[index] for index in top_indices]
        return select_list

    def update(self, candidate_id, arm_features, reward):
        # 更新A和b
        features = (arm_features * self.firm_features.T).reshape(-1)
        self.A += np.outer(features, features)
        self.b += reward * features
        self.theta = np.linalg.inv(self.A).dot(self.b)
        self.selected_candidates_feature.append(arm_features)

    def update_decentralized(self, arm_features):
        self.selected_candidates_feature.append(arm_features)

class GreedyFirm:
    def __init__(self, firm_id, firm_features, firm_features_length,candidate_feature_length, hired_num):
        self.firm_id = firm_id
        self.firm_features = firm_features
        self.candidate_feature_length = candidate_feature_length
        self.firm_features_length = firm_features_length
        self.alpha = 1.0
        self.lambda_reg = 1.0
        self.A = self.lambda_reg * np.identity(self.candidate_feature_length * self.firm_features_length)
        self.b = np.zeros(self.candidate_feature_length * self.firm_features_length)
        # self.theta_matrix= np.zeros((firm_features_length,candidate_feature_length), dtype=float)
        self.theta_matrix= np.full((firm_features_length,candidate_feature_length), 1)
        self.theta = self.theta_matrix.flatten()
        self.selected_nums = np.zeros((firm_features_length,candidate_feature_length))
        self.current_candidate_feature_vector = None
        self.current_candidate_group = None
        self.selected_candidates_feature = []
        self.hired_num = hired_num
        self.epislon = 0.1
        self.last_choice = -1
    

    def select_arm(self):
        if np.random.rand() < self.epislon:
            # select_indices = (self.last_choice + 1) // len(self.current_candidate_group)
            # self.last_choice = select_indices
            select_indices = np.random.choice(len(self.current_candidate_group))
            self.selected_candidates_feature.append(self.current_candidate_feature_vector[select_indices])
            return [self.current_candidate_group[select_indices]]
        new_features = [(feat * self.firm_features.T).reshape(-1) for feat in self.current_candidate_feature_vector]
        sampled_values = [feat.dot(self.theta) for feat in new_features]
        top_indices = [random_argmax(np.array(sampled_values))]
        select_list = [self.current_candidate_group[index] for index in top_indices]
        self.selected_candidates_feature.append(self.current_candidate_feature_vector[top_indices[0]])

        return select_list

    def select_arm_decentralized(self, theta, A):
        if np.random.rand() < self.epislon:
            # select_indices = (self.last_choice + 1) // len(self.current_candidate_group)
            # self.last_choice = select_indices
            select_indices = np.random.choice(len(self.current_candidate_group))
            self.selected_candidates_feature.append(self.current_candidate_feature_vector[select_indices])
            return [self.current_candidate_group[select_indices]]
        new_features = [(feat * self.firm_features.T).reshape(-1) for feat in self.current_candidate_feature_vector]
        sampled_values = [feat.dot(theta) for feat in new_features]
        top_indices = [random_argmax(np.array(sampled_values))]
        select_list = [self.current_candidate_group[index] for index in top_indices]
        self.selected_candidates_feature.append(self.current_candidate_feature_vector[top_indices[0]])

        return select_list        

    def update(self,candidate_id, arm_features, reward):
        # Ridge Regression estimate theta
        # (A + λI)^(-1) * b
        features = (arm_features * self.firm_features.T).reshape(-1)
        self.A += np.outer(features, features)
        self.b += reward * features
        self.theta = np.linalg.inv(self.A).dot(self.b)
        # self.selected_candidates_feature.append(arm_features[0])

    def update_decentralized(self, arm_features):
        # self.selected_candidates_feature.append(arm_features[0])
        pass