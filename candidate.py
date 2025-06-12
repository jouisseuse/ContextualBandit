from utils import *
import numpy as np
from scipy.stats import beta

'''
code for the candidate
'''
# import logging
# logging.basicConfig(
#     filename='log/UCB-8012-v1.log',  # Name of the log file
#     level=logging.DEBUG,  # Logging level
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format of the log messages
#     datefmt='%Y-%m-%d %H:%M:%S'  # Date format
# )

class Candidates:
    def __init__(self):
        self.candidate_feature_template = load_feature_distributions("candidate.json")
        self.candidate_features = generate_all_feature_combinations(self.candidate_feature_template)
        self.candidates = self.generate_candidates()
        self.firm_feature_template = load_feature_distributions("firm.json")
        self.firm_features = generate_all_feature_combinations(self.firm_feature_template)
        self.firms_num = len(self.firm_features)
        self.firm_feature_length = len(self.firm_features[0])
        

    def generate_candidates(self):
        candidate_list = []
        i = 0
        for feature in self.candidate_features:
            feature_vector = encode_one_hot(feature,self.candidate_feature_template)
            new_candidate = Candidate(i, self.firms_num, feature_vector, self.firm_feature_length, len(feature_vector))
            candidate_list.append(new_candidate)
            i=i+1
        
        return candidate_list
    
def random_argmax(list):
    return np.random.choice(np.flatnonzero(list == list.max()))

class Candidate:
    def __init__(self, candidate_id, firms, feature_vector,firm_feature_vector_length, candidate_feature_vector_length,firm_features):
        self.candidate_id = candidate_id
        self.firm_features = firm_features
        self.firms = firms # int, nums of firms
        self.feature_vector = feature_vector
        self.firm_feature_length = firm_feature_vector_length
        self.candidate_feature_length = candidate_feature_vector_length
        self.estimate_beta_a_matrix = np.ones((self.firm_feature_length, self.candidate_feature_length))
        self.estimate_beta_b_matrix = np.ones((self.firm_feature_length, self.candidate_feature_length))
        self.selected_firms = []
        self.applied_num = 1
        self.alpha = 1.0
        self.lambda_reg = 1.0

    def get_feature_vector(self):
        return self.feature_vector

    def get_id(self):
        return self.candidate_id
    
    
    def update_matrix(self, firm_id, rewards):
        for i in range(self.candidate_feature_length):
            if self.feature_vector[0][i] == 1:
                for j in range(self.firm_feature_length):
                    if self.firm_features[firm_id][0][j] > 0:
                        self.update_beta_matrix(j, i, rewards)

    def update_beta_matrix(self, rows, cols, rewards):
        if rewards == 1:
            self.estimate_beta_a_matrix[rows, cols] += 1
        self.estimate_beta_b_matrix[rows, cols] += 1

    def get_estimate_reward_distributed(self,firm_id):
        # return the expected reward of the ith firm
        # sum the beta sample of rows ith
        sum_reward = 0
        for i in range(self.candidate_feature_length):
            if self.feature_vector[0][i] == 1:
                for k in range(self.firm_feature_length):
                    if self.firm_features[firm_id][0][k] > 0:
                        sum_reward += np.random.beta(self.estimate_beta_a_matrix[k][i], self.estimate_beta_b_matrix[k][i]) * self.firm_features[firm_id][0][k]
        
        return sum_reward
    
    def select_firm_distributed(self):
        expected_rewards = [self.get_estimate_reward_distributed(i) for i in range(self.firms)]
        expected_rewards = np.array(expected_rewards)
        # selected_firm = expected_rewards.argmax()
        selected_firm = random_argmax(expected_rewards)
        self.selected_firms.append(selected_firm)
        
        return selected_firm
    
    def get_estimate_reward_decentralized(self,j, estimate_beta_a_matrix, estimate_beta_b_matrix):
        # return the expected reward of the ith firm
        # sum the beta sample of rows ith
        sum_reward = 0
        for i in range(self.candidate_feature_length):
            if self.feature_vector[0][i] == 1:
                for k in range(self.firm_feature_length):
                    if self.firm_features[j][0][k] > 0:
                        sum_reward += np.random.beta(estimate_beta_a_matrix[k][i], estimate_beta_b_matrix[k][i]) * self.firm_features[j][0][k]
                # sum_reward += estimate_beta_a_matrix[j][i] /estimate_beta_b_matrix[j][i]
                
        return sum_reward
    
    def select_firm_decentralized(self, estimate_beta_a_matrix, estimate_beta_b_matrix):
        # print(self.estimate_beta_a_matrix)
        # print(self.estimate_beta_b_matrix)
        expected_rewards = [self.get_estimate_reward_decentralized(i,estimate_beta_a_matrix, estimate_beta_b_matrix) for i in range(self.firms)]
        # print(expected_rewards)
        expected_rewards = np.array(expected_rewards)
        # selected_firm = expected_rewards.argmax()
        selected_firm = random_argmax(expected_rewards)
        self.selected_firms.append(selected_firm)

        return selected_firm
    
    def select_firm_theta(self, theta, A):
        new_features = [(feat * self.feature_vector.T).reshape(-1) for feat in self.firm_features]
        UCB_values = [feat.dot(theta) + self.alpha * np.sqrt(feat.dot(np.linalg.inv(A)).dot(feat)) for feat in new_features]
        expected_rewards = np.array(UCB_values)
        selected_firm = random_argmax(expected_rewards)
        self.selected_firms.append(selected_firm)

        return selected_firm
     
class GreedyFirmUCandidate:
    def __init__(self, candidate_id, firms, feature_vector,firm_feature_vector_length, candidate_feature_vector_length,firm_features):
        self.candidate_id = candidate_id
        self.firm_features = firm_features
        self.firms = firms # int, nums of firms
        self.feature_vector = feature_vector
        self.firm_feature_length = firm_feature_vector_length
        self.candidate_feature_length = candidate_feature_vector_length
        self.estimate_beta_a_matrix = np.ones((self.firms, self.candidate_feature_length))
        self.estimate_beta_b_matrix = np.ones((self.firms, self.candidate_feature_length))
        # self.estimate_beta_a_matrix = np.full((self.firms, self.candidate_feature_length),9)
        # self.estimate_beta_b_matrix = np.full((self.firms, self.candidate_feature_length),9)
        self.selected_firms = []
        self.alpha = 1.0
        self.epsilon = 0.1

    def create_matrices(self):
        matrices = []
        for _ in range(self.firms):
            matrix = np.ones((self.firm_feature_length, self.candidate_feature_length))
            matrices.append(matrix)
        return matrices
    
    def get_feature_vector(self):
        return self.feature_vector

    def get_id(self):
        return self.candidate_id
    
    def update_matrix(self, firm_id, rewards):
        for i in range(self.candidate_feature_length):
            if self.feature_vector[0][i] == 1:
                if rewards == 1:
                    self.estimate_beta_a_matrix[firm_id][i] += 1
                else:
                    self.estimate_beta_b_matrix[firm_id][i] += 1

        # if rewards == 1:
        #     self.estimate_beta_a_matrix[firm_id][0] += 1
        # self.estimate_beta_b_matrix[firm_id][0] += 1


    def update_beta_matrix(self, firm_id, rows, cols, rewards):
        if rewards == 1:
            self.estimate_beta_a_matrix[firm_id][rows, cols] += 1
        else:
            self.estimate_beta_b_matrix[firm_id][rows, cols] += 1

    # def update_matrix(self, firm_id, rewards):
    #     for i in range(self.candidate_feature_length):
    #         if self.feature_vector[0][i] == 1:
    #             self.update_beta_matrix(firm_id, i, rewards)

    # def update_beta_matrix(self, rows, cols, rewards):
    #     if rewards == 1:
    #         self.estimate_beta_a_matrix[rows, cols] += 1
    #     self.estimate_beta_b_matrix[rows, cols] += 1
    
    def get_estimate_reward_distributed(self,firm_id):
        # return the expected reward of the ith firm
        # sum the beta sample of rows ith
        sum_reward = 0
        for i in range(self.candidate_feature_length):
            if self.feature_vector[0][i] == 1:
                sum_reward += self.estimate_beta_a_matrix[firm_id][i] / (self.estimate_beta_a_matrix[firm_id][i] + self.estimate_beta_b_matrix[firm_id][i])
        
        return sum_reward
    
    def select_firm_distributed(self):
        if np.random.random() < self.epsilon:
            selected_firm = np.random.randint(self.firms)
            self.selected_firms.append(selected_firm)
            return selected_firm
        # sampled = np.random.beta(self.estimate_beta_a_matrix, self.estimate_beta_b_matrix)
        # print(sampled)
        # expected_rewards = [np.dot(sampled[i], self.feature_vector[0].T) for i in range(self.firms)]
        expected_rewards = [self.get_estimate_reward_distributed(i) for i in range(self.firms)]
        expected_rewards = np.array(expected_rewards)
        # selected_firm = expected_rewards.argmax()
        selected_firm = random_argmax(expected_rewards)
        self.selected_firms.append(selected_firm)
        
        return selected_firm
    
    def get_estimate_reward_decentralized(self,j, estimate_beta_a_matrix, estimate_beta_b_matrix):
        # return the expected reward of the ith firm
        # sum the beta sample of rows ith
        sum_reward = 0
        for i in range(self.candidate_feature_length):
            if self.feature_vector[0][i] == 1:
                sum_reward += estimate_beta_a_matrix[i] / (estimate_beta_a_matrix[i] + estimate_beta_b_matrix[i])
                
        return sum_reward
    
    def select_firm_decentralized(self, estimate_beta_a_matrix, estimate_beta_b_matrix):
        if np.random.random() < self.epsilon:
            selected_firm = np.random.randint(self.firms)
            self.selected_firms.append(selected_firm)
            return selected_firm
        # print(self.estimate_beta_a_matrix)
        # print(self.estimate_beta_b_matrix)
        expected_rewards = [self.get_estimate_reward_decentralized(i,estimate_beta_a_matrix[i], estimate_beta_b_matrix[i]) for i in range(self.firms)]
        # print(expected_rewards)
        expected_rewards = np.array(expected_rewards)
        # selected_firm = expected_rewards.argmax()
        selected_firm = random_argmax(expected_rewards)
        self.selected_firms.append(selected_firm)

        return selected_firm
    
    def select_firm_theta(self, theta, A):
        new_features = [(feat * self.feature_vector.T).reshape(-1) for feat in self.firm_features]
        UCB_values = [feat.dot(theta[i]) + self.alpha * np.sqrt(feat.dot(np.linalg.inv(A[i])).dot(feat)) for i,feat in enumerate(new_features)]
        expected_rewards = np.array(UCB_values)
        selected_firm = random_argmax(expected_rewards)
        self.selected_firms.append(selected_firm)

        return selected_firm