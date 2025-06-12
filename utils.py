import json
import numpy as np
import itertools
from scipy.stats import entropy
from collections import Counter

def load_feature_distributions(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def generate_all_feature_combinations(features):
    """ Generate all possible combinations of feature values. """
    all_combinations = []
    for feature, options in features.items():
        all_combinations.append([(feature, option) for option in options])
    # Create Cartesian product of all feature options
    return [dict(combination) for combination in itertools.product(*all_combinations)]


def generate_feature_vector(features):
    vector = {}
    for feature, options in features.items():
        total = sum(options.values())
        probabilities = [count / total for count in options.values()]
        choices = list(options.keys())
        selected_option = np.random.choice(choices, p=probabilities)
        vector[feature] = selected_option
    return vector

def encode_one_hot(features, feature_template):
    one_hot_vector = []
    for feature, options in feature_template.items():
        for option in options:
            one_hot_vector.append(1 if option == features[feature] else 0)
    return np.array([one_hot_vector])

def cal_gini(features):
    if isinstance(features, np.float64):
        return 0.0,0.0,0.0
    # calculate gini index
    ### first: 0-1 cols
    n_total = features[0]+features[1]
    gini_index_sex = 1 - (features[0]/n_total)**2 - (features[1]/n_total)**2
    ### second: 2-7 cols
    n_total = features[2]+features[3]+features[4]+features[5]+features[6] #+features[7]
    gini_index_race = 1 - (features[2]/n_total)**2 - (features[3]/n_total)**2 - (features[4]/n_total)**2 - (features[5]/n_total)**2 - (features[6]/n_total)**2 #- (features[7]/n_total)**2
    ### third: 8-12 cols
    # n_total = features[8]+features[9]+features[10]+features[11]+features[12]
    gini_index_age = 0.0  # 1 - (features[8]/n_total)**2 - (features[9]/n_total)**2 - (features[10]/n_total)**2 - (features[11]/n_total)**2-(features[12]/n_total)**2
    print(f"Gini index for sex: {gini_index_sex}, for race: {gini_index_race}, for age: {gini_index_age}")

    return gini_index_sex, gini_index_race, gini_index_age

def cal_gini_bk(features):
    # calculate gini index
    ### first: 0-1 cols
    n_total = features[0]+features[1]
    gini_index_sex = 1 - (features[0]/n_total)**2 - (features[1]/n_total)**2
    ### second: 2-4 cols
    n_total = features[2]+features[3]+features[4]
    gini_index_race = 1 - (features[2]/n_total)**2 - (features[3]/n_total)**2 - (features[4]/n_total)**2
    ### third: 5-9 cols
    n_total = features[5]+features[6]+features[7]+features[8]+features[9]
    gini_index_age = 1 - (features[5]/n_total)**2 - (features[6]/n_total)**2 - (features[7]/n_total)**2 - (features[8]/n_total)**2 - (features[9]/n_total)**2
    print(f"Gini index for sex: {gini_index_sex}, for race: {gini_index_race}, for age: {gini_index_age}")


def cal_entropy(features):
    if len(features) < 2 :
        return 0.0
    data = np.array(features)
    gender = data[:, :2]
    race = data[:, 2:7]
    age = data[:, 8:13]

    gender_probs = np.mean(gender, axis=0)
    gender_entropy = entropy(gender_probs, base=2)

    race_probs = np.mean(race, axis=0)
    race_entropy = entropy(race_probs, base=2)

    age_probs = np.mean(age, axis=0)
    age_entropy = entropy(age_probs, base=2)
    age_entropy = 0.0

    total_entropy = gender_entropy + race_entropy + age_entropy

    return total_entropy

def calculate_entropy(vector_set):
    """
    计算给定向量集合的熵。
    
    参数:
    vector_set (list of tuple): 包含13维0/1向量的集合
    
    返回:
    float: 集合的熵值
    """
     # 将 numpy 数组转换为元组，因为元组是可哈希的
    tuple_vector_set = [tuple(vector) for vector in vector_set]
    
    # 计算每个向量出现的次数
    vector_counts = Counter(tuple_vector_set)
    
    # 总向量数
    total_vectors = len(vector_set)
    
    # 计算每个向量的概率
    probabilities = np.array(list(vector_counts.values())) / total_vectors
    
    # 使用 scipy 的 entropy 函数计算熵
    entropy_value = entropy(probabilities, base=2)  # base=2 表示以2为底的对数
    
    return entropy_value


### TEST ###
# data = load_feature_distributions("candidate.json")
# print(data)
# combinations = generate_all_feature_combinations(data)
# print(combinations)
# features2 = combinations[0]
# print(features2)
# one_hot2 = encode_one_hot(features2, data)
# print(one_hot2)
# features = generate_feature_vector(data)
# print(features)
# one_hot = encode_one_hot(features, data)
# print(one_hot)