from firm import GreedyFirm, UCBFirmMix, ThompsonSamplingFirmMix
from candidate import GreedyFirmUCandidate
from utils import *
from collections import Counter
import ast
import random
import numpy as np

from scipy.stats import entropy


class Environment:
    def __init__(self, round, hired_num):
        self.round = round
        self.mu_x = 1.5 # Mean of features
        self.sigma_x = 1  # Standard deviation of features
        self.sigma_epsilon = 0.1 # Standard deviation of noise
        self.candidate_feature_template = load_feature_distributions("candidate.json")
        self.candidate_features = generate_all_feature_combinations(self.candidate_feature_template)
        self.candidate_feature_length = 13
        self.firm_feature_template = load_feature_distributions("firm.json")
        # self.firm_features = generate_all_feature_combinations(self.firm_feature_template)
        self.firm_features = []
        self.firm_feature_length = 4
        self.hired_num = hired_num


        ### object
        self.firms = self.set_up_firms_from_txt()
        self.candidates = self.set_up_candidates()
        self.estimate_beta_a_matrix = self.create_matrices() 
        self.estimate_beta_b_matrix = self.create_matrices()

        ### for decentralized firm
        self.alpha = 1.0
        self.lambda_reg = 1.0
        self.theta_matrix= np.zeros((self.firm_feature_length, self.candidate_feature_length), dtype=float)
        # self.theta = [self.theta_matrix.flatten()] * 60
        # self.A = [self.lambda_reg * np.identity(self.candidate_feature_length * self.firm_feature_length)] * 60
        # self.b = [np.zeros(self.candidate_feature_length * self.firm_feature_length)] *60
        self.theta = self.theta_matrix.flatten()
        self.A = self.lambda_reg * np.identity(self.candidate_feature_length * self.firm_feature_length)
        self.b = np.zeros(self.candidate_feature_length * self.firm_feature_length)

    def create_matrices(self):
        matrices = []
        for _ in self.firms:
            matrix = np.ones((self.firm_feature_length, self.candidate_feature_length))
            matrices.append(matrix)
        return matrices

    def set_up_candidates(self):
        firm_features = [firm.firm_features for firm in self.firms]
        candidate_list = []
        i = 0
        for feature in self.candidate_features:
            feature_vector = encode_one_hot(feature, self.candidate_feature_template)
            # Greedy-GreedyFirmUCandidate
            new_candidate = GreedyFirmUCandidate(i, len(self.firm_features), feature_vector, self.firm_feature_length, self.candidate_feature_length, firm_features)
            candidate_list.append(new_candidate)
            i = i + 1
        
        return candidate_list
    
    
    def set_up_firms_from_txt(self):
        with open('firms_vector.txt', 'r') as f:
            list_from_file = [ast.literal_eval(line.replace(' ', ', ')) for line in f]
        firm_list = []
        i = 0
        for vector in list_from_file:
            firm_vector = np.array([vector])
            self.firm_features.append(firm_vector)
            new_firm = ThompsonSamplingFirmMix(i, firm_vector, self.firm_feature_length, self.candidate_feature_length, self.hired_num)
            firm_list.append(new_firm)
            i = i + 1
            # GreedyFirm,UCBFirmMix,ThompsonSamplingFirmMix


        return firm_list
    
    def get_current_arms_group(self):
        # get current arms' group id from self.arms and return a list of group ids
        return [arm.get_group() for arm in self.arms]
    
    def get_current_feature_vector(self):
        # get current arms' feature vector from self.arms and return a list of feature vectors
        return [arm.get_feature_vector() for arm in self.arms]
        
    def update_all_candidate_estimate(self,candidate_feature, firm_id, rewards):
        for i in range(self.candidate_feature_length):
            if candidate_feature[0][i] == 1:
                self.update_beta_matrix(firm_id, i, rewards)
                # for j in range(self.firm_feature_length):
                #     if self.firms[firm_id].firm_features[0][j] > 0:
                #         self.update_beta_matrix(j, i, rewards)
                #         # self.update_beta_matrix_individual(firm_id, j, i, rewards)

    def update_beta_matrix_individual(self, firm_id, rows, cols, rewards):
        if rewards:
            self.estimate_beta_a_matrix[firm_id][rows, cols] += 1
        else:
            self.estimate_beta_b_matrix[firm_id][rows, cols] += 1

    def update_all_candidates(self):
        for candidate in self.candidates:
            candidate.update_matrixs(self.estimate_beta_a_matrix, self.estimate_beta_b_matrix)

    def update_beta_matrix(self, rows, cols, rewards):
        if rewards == 1:
            self.estimate_beta_a_matrix[rows][cols] += 1
        else:
            self.estimate_beta_b_matrix[rows][cols] += 1

    def update_all_theta(self, arm_features, firm_features, reward):
        # Ridge Regression estimate theta
        # (A + λI)^(-1) * b
        features = (arm_features * firm_features.T).reshape(-1)
        self.A += np.outer(features, features)
        self.b += reward * features
        self.theta = np.linalg.inv(self.A).dot(self.b)

    def update_all_theta_ind(self, arm_id, arm_features, firm_features, reward):
        # Ridge Regression estimate theta
        # (A + λI)^(-1) * b
        features = (arm_features * firm_features.T).reshape(-1)
        self.A[arm_id] += np.outer(features, features)
        self.b[arm_id] += reward * features
        self.theta[arm_id] = np.linalg.inv(self.A[arm_id]).dot(self.b[arm_id])

    def run_simulation_distributedCandidate_unique(self):
        all_application_features = {firm.firm_id: [] for firm in self.firms}
    
        # Round 1 firms hire candidates, and get the reward
        initial_selection = list(range(0, 60))
        print(initial_selection)
        ## the firm hires given candidates
        for hired in range(self.hired_num):
            for firm in self.firms:
                selected_candidates = self.candidates[initial_selection[firm.firm_id]]
                theta = np.full((self.firm_feature_length, self.candidate_feature_length), 0.5)
                # theta = np.random.normal(self.mu_x, self.sigma_x, (self.firm_feature_length, self.candidate_feature_length))
                print("firm: ", firm.firm_id, "selected_features ", matrix_print(self.candidates[initial_selection[firm.firm_id]].feature_vector[0]))
                # print(theta)
                # print(firm.firm_features)
                reward = np.sum(theta * selected_candidates.feature_vector * firm.firm_features.T) + np.random.normal(0,self.sigma_epsilon)
                firm.update(selected_candidates.candidate_id, selected_candidates.feature_vector[0],reward)

                ### two-sided
                # # update candidate beta matrix
                # ### distributed
                # for candidate in self.candidates:
                #     if candidate.candidate_id == selected_candidates.candidate_id:
                #         candidate.update_matrix(firm.firm_id, 1)
                #     else:
                #         candidate.update_matrix(firm.firm_id, 0)
        
        print("==================== Game Start =======================")
        for rd in range(self.round):
            print("--------------------------------------------------------------------")
            print("Turn {} start".format(rd))
            ### two-sided
            # # Round 2 candidates apply for firms, firms hire candidates, and get the reward
            # applications = {firm.firm_id: [] for firm in self.firms}  # initialize applications
            # applications_features = {firm.firm_id: [] for firm in self.firms}  # initialize applications
            # for candidate in self.candidates:
            #     applied_firms = candidate.select_firm_distributed()
            #     applications[applied_firms].append(candidate.candidate_id)
            #     applications_features[applied_firms].append(candidate.feature_vector[0])
            #     all_application_features[applied_firms].append(candidate.feature_vector[0])

            ### one-sided
            applications = {firm.firm_id: [] for firm in self.firms}  # initialize applications
            applications_features = {firm.firm_id: [] for firm in self.firms}  # initialize applications
            for candidate in self.candidates:
                for firm in self.firms:
                    applications[firm.firm_id].append(candidate.candidate_id)
                    applications_features[firm.firm_id].append(candidate.feature_vector[0])
                    all_application_features[firm.firm_id].append(candidate.feature_vector[0])
            # print("applications:",applications)

            # Round 3 firms hire workers, and get the reward
            offers = {firm.firm_id: [] for firm in self.firms}  # initialize offers
            for firm in self.firms:
                if applications[firm.firm_id] != []:
                    firm.current_candidate_feature_vector = applications_features[firm.firm_id]
                    firm.current_candidate_group = applications[firm.firm_id]
                    offered_candidate = firm.select_arm()
                    # for candidate in offered_candidate:
                    offers[firm.firm_id] = offered_candidate
                else:
                    print("firm:",firm.firm_id,"no applications")
            
            ## calculate candidate's selected firms
            # for firm_id, applications_features in applications_features.items():
            #     print("firm: ", firm_id, "applied_features ", matrix_print(np.sum(applications_features, axis=0)))
            
            # Round 4 update rewards
            # for candidates, use hire to update beta parameters
            for firm_id, candidate_list in applications.items():
                ### distributed
                for candidate_id in candidate_list:
                    candidate = self.candidates[candidate_id]
                    if candidate_id in offers[firm_id]:
                        # candidate.update_matrix(firm_id, 1)
                        # update firm
                        theta = np.full((self.firm_feature_length, self.candidate_feature_length), 0.5)
                        firm = self.firms[firm_id]
                        reward = np.sum(theta * candidate.feature_vector * firm.firm_features.T) + np.random.normal(0,self.sigma_epsilon)
                        firm.update(candidate.candidate_id, candidate.feature_vector[0], reward)
                    else:
                        # candidate.update_matrix(firm_id, 0)
                        pass
                    
        print("==================== Game End =======================")
        print("-------------------Offer Results---------------------------------")
        ## calculate firm's selected features
        for firm in self.firms:
            print("firm: ", firm.firm_id, "initsel_features ", matrix_print(self.candidates[initial_selection[firm.firm_id]].feature_vector[0]))
            print("firm: ", firm.firm_id, "select_features  ", matrix_print(np.sum(firm.selected_candidates_feature, axis=0)))
            print("--------------------------------------------------------------------")

        # print("-------------------Application Results------------------------------")
        # for firm_id, applications_features in all_application_features.items():
        #     print("firm: ", firm_id, "initsel_features ", matrix_print(self.candidates[initial_selection[firm_id]].feature_vector[0]))
        #     print("firm: ", firm_id, "applied_features ", matrix_print(np.sum(applications_features, axis=0)))
        #     print("--------------------------------------------------------------------")
        
    def run_simulation_decentralizedCandidate_unique(self):
        all_application_features = {firm.firm_id: [] for firm in self.firms}
    
        # Round 1 candidates apply for firms, firms hire candidates, and get the reward
        initial_selection = list(range(0,60))
        ## the firm hires given candidates
        for hired in range(self.hired_num):
            for firm in self.firms:
                selected_candidates = self.candidates[initial_selection[firm.firm_id]]
                theta = np.full((self.firm_feature_length, self.candidate_feature_length), 0.5)
                print("firm: ", firm.firm_id, "selected_features ", matrix_print(self.candidates[initial_selection[firm.firm_id]].feature_vector[0]))
                reward = np.sum(theta * selected_candidates.feature_vector * firm.firm_features.T) + 1.0 #+ np.random.normal(0,self.sigma_epsilon)
                firm.update(selected_candidates.candidate_id,selected_candidates.feature_vector[0],reward)

                # update candidate beta matrix
                ### two-sided
                ### decentralized
                # for candidate in self.candidates:
                #     if candidate.candidate_id == selected_candidates.candidate_id:
                #         self.update_all_candidate_estimate(candidate.feature_vector, firm.firm_id, 1)
                #     else:
                #         self.update_all_candidate_estimate(candidate.feature_vector, firm.firm_id, 0)
      
        print("==================== Game Start =======================")
        for rd in range(self.round):
            print("--------------------------------------------------------------------")
            print("Turn {} start".format(rd))
            ### two-sided
            # # Round 2 candidates apply for firms, firms hire candidates, and get the reward
            # applications = {firm.firm_id: [] for firm in self.firms}  # initialize applications
            # applications_features = {firm.firm_id: [] for firm in self.firms}  # initialize applications
            # for candidate in self.candidates:
            #     applied_firms = candidate.select_firm_decentralized(self.estimate_beta_a_matrix,self.estimate_beta_b_matrix)
            #     applications[applied_firms].append(candidate.candidate_id)
            #     applications_features[applied_firms].append(candidate.feature_vector[0])
            #     all_application_features[applied_firms].append(candidate.feature_vector[0])
            
            ### one-sided
            applications = {firm.firm_id: [] for firm in self.firms}  # initialize applications
            applications_features = {firm.firm_id: [] for firm in self.firms}  # initialize applications
            for candidate in self.candidates:
                for firm in self.firms:
                    applications[firm.firm_id].append(candidate.candidate_id)
                    applications_features[firm.firm_id].append(candidate.feature_vector[0])
                    all_application_features[firm.firm_id].append(candidate.feature_vector[0])
            # print("applications:",applications)

            # Round 3 firms hire workers, and get the reward
            offers = {firm.firm_id: [] for firm in self.firms}  # initialize offers
            for firm in self.firms:
                if applications[firm.firm_id] != []:
                    firm.current_candidate_feature_vector = applications_features[firm.firm_id]
                    firm.current_candidate_group = applications[firm.firm_id]
                    offered_candidate = firm.select_arm()
                    offers[firm.firm_id] = offered_candidate
                else:
                    print("firm:",firm.firm_id,"no applications")
            
            ## calculate candidate's selected firms
            # for firm_id, applications_features in applications_features.items():
            #     print("firm: ", firm_id, "applied_features ", matrix_print(np.sum(applications_features, axis=0)))
            
            # Round 4 update rewards
            # for candidates, use hire to update beta parameters
            for firm_id, candidate_list in applications.items():
                ### decentralized
                for candidate_id in candidate_list:
                    candidate = self.candidates[candidate_id]
                    if candidate_id in offers[firm_id]:
                        # self.update_all_candidate_estimate(candidate.feature_vector, firm_id, 1)
                        # update firm
                        theta = np.full((self.firm_feature_length, self.candidate_feature_length), 0.5)
                        firm = self.firms[firm_id]
                        reward = np.sum(theta * candidate.feature_vector * firm.firm_features.T) + np.random.normal(0,self.sigma_epsilon)
                        firm.update(candidate.candidate_id, candidate.feature_vector[0], reward)
                    else:
                        # self.update_all_candidate_estimate(candidate.feature_vector, firm_id, 0)
                        pass

        print("==================== Game End =======================")
        # for candidate in self.candidates:
        #     firms_counters = Counter(candidate.selected_firms)
        #     for firm_id, count in firms_counters.items():
        #         print(f"candidate {candidate.candidate_id} select firm {firm_id} for {count} times")
        print("-------------------Offer Results---------------------------------")
        ## calculate firm's selected features
        for firm in self.firms:
            # print("firm theta:", firm.firm_id, firm.theta)
            print("firm: ", firm.firm_id, "initsel_features ", matrix_print(self.candidates[initial_selection[firm.firm_id]].feature_vector[0]))
            print("firm: ", firm.firm_id, "select_features  ", matrix_print(np.sum(firm.selected_candidates_feature, axis=0)))
            print("--------------------------------------------------------------------")

        # print("-------------------Application Results------------------------------")
        # for firm_id, applications_features in all_application_features.items():
        #     print("firm: ", firm_id, "initsel_features ", matrix_print(self.candidates[initial_selection[firm_id]].feature_vector[0]))
        #     print("firm: ", firm_id, "applied_features ", matrix_print(np.sum(applications_features, axis=0)))
        #     print("--------------------------------------------------------------------")

def matrix_print(matrix):
    if isinstance(matrix, np.float64):
        matrix = [0] * 13
    elif not hasattr(matrix, '__iter__'):
        matrix = [matrix]
    formatted_matrix =  np.array([f"{num:2d}" for num in matrix]) 
    return formatted_matrix

### code for test ###
env = Environment(2, 1)
# env.run_simulation_distributedCandidate_unique()
env.run_simulation_decentralizedCandidate_unique()