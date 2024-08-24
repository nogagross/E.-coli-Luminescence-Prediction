import pandas as pd
import ast
from Bio import motifs
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
import seqfold
from seqfold import dg
import numpy as np
from Bio.SeqUtils import MeltingTemp as mt
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
def str_to_list(seq):
  numbers = [float(num) for num in ast.literal_eval(seq)]
  return numbers


def calculate_energy(sequence, df):
    sequence_length = len(sequence)
    energies = []

    for i in range(sequence_length - 5):  # Stop at sequence_length - 5 to ensure window of 6 codons
        codon_segment = sequence[i:i + 6]

        energy_value = df[df['Sequence'] == codon_segment]['Energy'].values[0]
        energies.append(float(energy_value))
    return energies


def find_mutations(variant_sequence, control_sequence):
    mutations = []
    for i, (variant_nuc, control_nuc) in enumerate(zip(variant_sequence, control_sequence)):
        if variant_nuc != control_nuc:
            mutations.append((i, control_nuc, variant_nuc))
    return mutations

# Function to create a string with changed nucleotides
def get_changed_nucleotides(mutations):
    return ''.join([variant_nuc for _, _, variant_nuc in mutations])

# Function to create a list with indices of the changed nucleotides
def get_mutation_indices(mutations):
    return [pos for pos, _, _ in mutations]
#get sequnce of changed nucoutides
def get_changed_nucleotides_original(mutations):
    return ''.join([control_nuc for _, control_nuc, _ in mutations])


def align_sequences(seq1, seq2):
  aligner1 = PairwiseAligner()
  aligner1.substitution_matrix = substitution_matrices.load("NUC.4.4")
  align1 = aligner1.align(seq1, seq2)
  return align1.score

def vectorized_score(seq):
    score = align_sequences(seq[1:-1], control_var['Variant sequence'].values[0].strip("'"))
    return score

def set_difference(list1):
  return list(set(list1) - set(control_var['diff_energy_fold'].values[0]))

def calculate_mean(x):
  if len(x) > 0:
    return sum(x) / len(x)
  else:
    return 0
def calculate_energy_folding(sequence):
    sequence_length = len(sequence)
    energies = []

    for i in range(sequence_length - 39):
        codon_segment = sequence[i:i + 40]

        energy_value = dg(codon_segment)
        energies.append(float(energy_value))
    return energies

def mean_windows_folding(windows):
  num_features = len(windows)//20
  means = []
  for i in range(num_features):
    start_idx = i * 20
    end_idx = (i + 1) * 20
    mean_of_windows = np.mean(windows[start_idx:end_idx])
    means.append(mean_of_windows)
  return means

def calculate_pssm_score(sequence, pssm):
    sequence = sequence.strip("'")
    max_score = float('-inf')
    max_index = -1
    for i in range(len(sequence) - len(pssm) + 1):
        subseq = sequence[i:i+len(pssm)]
        score = 1
        for j, base in enumerate(subseq):
            score *= pssm.loc[base, j+1]
        if score > max_score:
            max_score = score
            max_index = i
    return np.log(max_score), max_index



def compare_codons_flexible(variant_sequence, control_sequence):

    def get_codons(sequence):
        sequence = sequence.strip("'")
        return [sequence[i:i+3] for i in range(0, len(sequence) - 2, 3)]

    variant_codons = get_codons(variant_sequence)
    control_codons = get_codons(control_sequence)

    # Compare codons
    changed_codons = 0


    for i in range(min(len(variant_codons), len(control_codons))):
        var_codon = variant_codons[i]
        ctrl_codon = control_codons[i]
        if len(var_codon) == 3 and len(ctrl_codon) == 3:
            if var_codon != ctrl_codon:
                changed_codons += 1
    return changed_codons

def calculate_tm(seq):
    return mt.Tm_NN(Seq(seq))

def mutate_sequence(seq, position):
    bases = 'ATCG'
    original_base = seq[position]
    mutated_seqs = []
    for base in bases:
        if base != original_base:
            mutated_seq = seq[:position] + base + seq[position+1:]
            mutated_seqs.append(mutated_seq)
    return mutated_seqs

def calculate_robustness(seq):
    original_tm = calculate_tm(seq)
    tm_changes = []

    for i in range(len(seq)):
        mutated_seqs = mutate_sequence(seq, i)
        for mutated_seq in mutated_seqs:
            mutated_tm = calculate_tm(mutated_seq)
            tm_change = abs(mutated_tm - original_tm)
            tm_changes.append(tm_change)

    avg_tm_change = np.mean(tm_changes)
    robustness = 1 / avg_tm_change if avg_tm_change != 0 else float('inf')

    return robustness




def spearman_scorer(y_true, y_pred):
    # Convert inputs to DataFrame if they are not already
    y_true = pd.DataFrame(y_true)
    y_pred = pd.DataFrame(y_pred)

    # Compute Spearman correlation for each column
    spearman_scores = [spearmanr(y_true.iloc[:, i], y_pred.iloc[:, i]).statistic for i in range(y_true.shape[1])]

    # Return the mean of Spearman correlations
    return np.mean(spearman_scores)

df_var_train = pd.read_excel('Train_data(1).xlsx')
df_with_dnt_train = pd.read_excel('Train_data(1).xlsx', sheet_name="luminescence with DNT")
df_without_dnt_train = pd.read_excel('Train_data(1).xlsx', sheet_name="luminescence without DNT")
df_features = pd.read_excel('Train_data(1).xlsx', sheet_name="Features")

df_var_test = pd.read_excel('Test_data (3).xlsx',)
df_test_features = pd.read_excel('Test_data (3).xlsx', sheet_name="Features")

#sort in acsending order train_data
df_var_train.sort_values(by='Variant number', inplace = True)
df_with_dnt_train.sort_values(by='Variant number', inplace =True)
df_without_dnt_train.sort_values(by='Variant number', inplace = True)

#defining the target
diff = df_with_dnt_train - df_without_dnt_train

y_train3 = pd.DataFrame()
y_train3['max_diff'] = diff.max(axis=1)
y_train3['avg_diff'] = abs(diff.mean(axis=1))

y_train3.to_excel('y_train.xlsx')
#seperate the control variant from the original one
control_var = df_var_train[df_var_train['Variant number'] == 1]
df_var_train = df_var_train[df_var_train['Variant number'] != 1]
merged_df = pd.merge(df_var_train,df_features , on='Variant number', how='inner')


merged_df_test = pd.merge(df_var_test,df_test_features , on='Variant number', how='inner')

#df_hb_energy = pd.DataFrame
#df_var_train['Sequence'] = df_var_train['Variant Sequence'].apply(lambda x:x[1:-1])
#df_var_train['asd'] = df_var_train['Sequence'].apply(calculate_energy)
#df_var_test['Sequence'] = df_var_test['Variant Sequence'].apply(lambda x:x[1:-1])
#df_var_test['asd'] = df_var_train['Sequence'].apply(calculate_energy)
#df_var_train.drop(['Sequence'],inplace = True)
#df_var_test.drop(['Sequence'],inplace = True)

#df_var_train.to_csv('train_asd_df.csv')
#df_var_test.to_csv('test_asd_df.csv')

#df_var_train.drop(['asd'],inplace = True)
#df_var_test.drop(['asd'],inplace = True)

merged_df["asd"] = pd.read_csv("train_asd_df.csv", usecols=["asd_enr"])
merged_df['asd'] = merged_df['asd'].apply(str_to_list)

merged_df_test["asd"] = pd.read_csv("test_asd-df.csv", usecols=["asd_enr"])
merged_df_test['asd'] = merged_df_test['asd'].apply(str_to_list)


#adding to the data frame the hibredization energy
merged_df['sum_asd'] =  merged_df['asd'].apply(lambda x:sum(x))
merged_df['min_asd'] =  merged_df['asd'].apply(lambda x:min(x))
merged_df['mean_asd'] =  merged_df['asd'].apply(lambda x:sum(x)/len(x))


#adding to the  test data frame the hibredization energy
merged_df_test['sum_asd'] =  merged_df['asd'].apply(lambda x:sum(x))
merged_df_test['min_asd'] =  merged_df['asd'].apply(lambda x:min(x))
merged_df_test['mean_asd'] =  merged_df['asd'].apply(lambda x:sum(x)/len(x))




# Find mutations for each variant sequence compared to the control sequence
merged_df['Mutations'] = merged_df['Variant sequence'].apply(lambda x: find_mutations(x[1:-1],control_var['Variant sequence'].values[0].strip("'")))
merged_df_test['Mutations'] = merged_df['Variant sequence'].apply(lambda x: find_mutations(x[1:-1],control_var['Variant sequence'].values[0].strip("'")))



# Apply the functions to create the string and list for each variant
merged_df['Changed Nucleotides'] = merged_df['Mutations'].apply(get_changed_nucleotides)
merged_df['original Nucleotides'] = merged_df['Mutations'].apply(get_changed_nucleotides_original)
merged_df['Mutation Indices'] = merged_df['Mutations'].apply(get_mutation_indices)


merged_df_test['Changed Nucleotides'] = merged_df['Mutations'].apply(get_changed_nucleotides)
merged_df_test['original Nucleotides'] = merged_df['Mutations'].apply(get_changed_nucleotides_original)
merged_df_test['Mutation Indices'] = merged_df['Mutations'].apply(get_mutation_indices)

merged_df['len_mutation'] = merged_df['Mutation Indices'].apply(lambda x:len(x))
merged_df_test['len_mutation'] = merged_df_test['Mutation Indices'].apply(lambda x:len(x))





merged_df['alignment_score'] = merged_df['Variant sequence'].apply(vectorized_score)
merged_df_test['alignment_score'] = merged_df_test['Variant sequence'].apply(vectorized_score)





#df_var_test['fold_var'] = df_var_test['Variant sequence'].apply(calculate_energy_folding)
#df_var_test.to_csv('test_folding_energy-df.csv')

#merged_df['fold_var'] = merged_df['Variant sequence'].apply(calculate_energy_folding)
#merged_df.to_csv('temp_merged-df.csv')


# Read the CSV file with the column you want to extract
new_column_data = pd.read_csv("temp_merged-df.csv", usecols=["folding_energy_window"])
new_column_data_test = pd.read_csv("test_folding_energy-df.csv", usecols=["fold_var"])


# Add the extracted column to your DataFrame
merged_df["folding_energy_window"] = new_column_data
merged_df['folding_energy_window'] = merged_df['folding_energy_window'].apply(str_to_list)

merged_df_test["folding_energy_window"] = new_column_data_test
merged_df_test['folding_energy_window'] = merged_df_test['folding_energy_window'].apply(str_to_list)

#find the minimum and the mean folding energy window
merged_df['folding_energy_window_min'] = merged_df['folding_energy_window'].apply(lambda x:min(x))
merged_df['folding_energy_window_mean'] = merged_df['folding_energy_window'].apply(lambda x:sum(x)/len(x))

merged_df_test['folding_energy_window_min'] = merged_df_test['folding_energy_window'].apply(lambda x:min(x))
merged_df_test['folding_energy_window_mean'] = merged_df_test['folding_energy_window'].apply(lambda x:sum(x)/len(x))



#find the mean diffrance between the folding energy of the control and the variat
control_var['diff_energy_fold'] = control_var['Variant sequence'].apply(lambda x:calculate_energy_folding(x[1:-1]))


merged_df['diff_con_var_energy_fold'] = merged_df['folding_energy_window'].apply(set_difference)
merged_df['diff_con_var_energy_fold_mean'] = merged_df['diff_con_var_energy_fold'].apply(calculate_mean)

merged_df_test['diff_con_var_energy_fold'] = merged_df_test['folding_energy_window'].apply(set_difference)
merged_df_test['diff_con_var_energy_fold_mean'] = merged_df_test['diff_con_var_energy_fold'].apply(calculate_mean)


merged_df['means_fold_energies'] = merged_df['folding_energy_window'].apply(mean_windows_folding)
merged_df_test['means_fold_energies'] = merged_df_test['folding_energy_window'].apply(mean_windows_folding)

num_features = len(merged_df['folding_energy_window'][0])//20


#mean 20th folding energy windows
for i in range(0,num_features):
  merged_df[f'folding_energy_mean_window_{i}'] = merged_df['means_fold_energies'].apply(lambda x: x[i])
  merged_df_test[f'folding_energy_mean_window_{i}'] = merged_df_test['means_fold_energies'].apply(lambda x: x[i])


#pssm
pssm_xls = pd.ExcelFile('PSSM (1).xlsx')
pssm_dict = {}
for sheet_name in pssm_xls.sheet_names:
    pssm_dict[sheet_name] = pssm_xls.parse(sheet_name, index_col=0)

list_of_pssms=[pssm_dict['Motif 1'],pssm_dict['Motif 2'],pssm_dict['Motif 3'],pssm_dict['Motif 4'],pssm_dict['Motif 5'],pssm_dict['Motif 6'],pssm_dict['Motif 7'],pssm_dict['Motif 8'],pssm_dict['Motif 9'],pssm_dict['Motif 11'],pssm_dict['Motif 12']]



#find maximum pssm and the start index for the maximum motif
for name, pssm in pssm_dict.items():
    scores_and_indices = merged_df['Variant sequence'].apply(lambda x: calculate_pssm_score(x, pssm))
    merged_df[f'PSSM_{name}'] = scores_and_indices.apply(lambda x: x[0])  # Max PSSM score
    merged_df[f'PSSM_{name}_index'] = scores_and_indices.apply(lambda x: x[1])  # Index of
    scores_and_indices_test = merged_df['Variant sequence'].apply(lambda x: calculate_pssm_score(x, pssm))
    merged_df_test[f'PSSM_{name}'] = scores_and_indices_test.apply(lambda x: x[0])  # Max PSSM score
    merged_df_test[f'PSSM_{name}_index'] = scores_and_indices_test.apply(lambda x: x[1])  # Index of








merged_df['changed codons'] = merged_df.apply(lambda row: compare_codons_flexible(row['Variant sequence'], control_var['Variant sequence'].values[0].strip("'")), axis=1)
merged_df_test['changed codons'] = merged_df_test.apply(lambda row: compare_codons_flexible(row['Variant sequence'], control_var['Variant sequence'].values[0].strip("'")), axis=1)

dinucleotides = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
for di in dinucleotides:
    merged_df[f'count_{di}'] = merged_df['Variant sequence'].apply(lambda x: x.count(di))
    merged_df_test[f'count_{di}'] = merged_df_test['Variant sequence'].apply(lambda x: x.count(di))





merged_df['Tm'] = merged_df['Variant sequence'].apply(calculate_tm)
merged_df['robustness'] = merged_df['Variant sequence'].apply(calculate_robustness)
merged_df_test['Tm'] = merged_df_test['Variant sequence'].apply(calculate_tm)
merged_df_test['robustness'] = merged_df_test['Variant sequence'].apply(calculate_robustness)







#remove nun numeric features
final_df= merged_df
final_df_test = merged_df_test
final_df.drop(columns=['Changed codons','Variant number','Variant sequence', 'Folding energy window 1','Folding energy window 2','asd','Changed Nucleotides','original Nucleotides','Mutation Indices','diff_con_var_energy_fold','Mutations','folding_energy_window','means_fold_energies'], inplace= True)
final_df_test.drop(columns=['Changed codons','Variant sequence', 'Folding energy window 1','Folding energy window 2','asd','Changed Nucleotides','original Nucleotides','Mutation Indices','diff_con_var_energy_fold','Mutations','folding_energy_window','means_fold_energies'], inplace= True)



scaler = StandardScaler()

df_standardized_array = scaler.fit_transform(final_df)
df_standardized_array_test = scaler.fit_transform(final_df_test)

df_standardized = pd.DataFrame(df_standardized_array, columns=final_df.columns)
df_standardized_test = pd.DataFrame(df_standardized_array_test, columns=final_df_test.columns)



class ForwardFeatureSelector_random_forest(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=5, cv=5):
        self.n_features = n_features
        self.selected_features = []
        self.cv = cv

    def fit(self, X, y):
        self.selected_features = []
        remaining_features = list(X.columns)

        while remaining_features:
            best_feature = None
            best_correlation = -np.inf

            for feature in remaining_features:
                features_to_test = self.selected_features + [feature]
                X_test = X[features_to_test]

                xgbmodel = xgb.XGBRegressor(objective='reg:squarederror')

                y_pred = cross_val_predict(xgbmodel, X_test, y, cv=self.cv)


                correlation = spearman_scorer(y,y_pred)

                if correlation > best_correlation:
                    best_correlation = correlation
                    best_feature = feature

            if best_feature is None:
                break

            self.selected_features.append(best_feature)
            remaining_features.remove(best_feature)

            if self.n_features is not None and len(self.selected_features) == self.n_features:
                break

        return self

    def transform(self, X):
        print(self.selected_features)
        return X[self.selected_features]

#selceting features for random forest
class ForwardFeatureSelector_random_forest(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=5, cv=5):
        self.n_features = n_features
        self.selected_features = []
        self.cv = cv

    def fit(self, X, y):
        self.selected_features = []
        remaining_features = list(X.columns)

        while remaining_features:
            best_feature = None
            best_correlation = -np.inf

            for feature in remaining_features:
                features_to_test = self.selected_features + [feature]
                X_test = X[features_to_test]

                # Build and evaluate the model
                model_forest = RandomForestRegressor(n_estimators=100, random_state=42)

                y_pred = cross_val_predict(model_forest, X_test, y, cv=self.cv)
                correlation = spearman_scorer(y,y_pred)

                if correlation > best_correlation:
                    best_correlation = correlation
                    best_feature = feature

            if best_feature is None:
                break

            self.selected_features.append(best_feature)
            remaining_features.remove(best_feature)

            if self.n_features is not None and len(self.selected_features) == self.n_features:
                break

        return self

    def transform(self, X):
        print(self.selected_features)
        return X[self.selected_features]


selector_forest = ForwardFeatureSelector_random_forest(n_features=5, cv=5)  # Select top 5 features with 5-fold cross-validation
X_train_selected_forest = selector_forest.fit_transform(df_standardized, y_train3)
X_test_selected_forest = selector_forest.transform(df_standardized_test.iloc[:,1:])
X_test_selected_forest_with_names = pd.concat([df_var_test,X_test_selected_forest],axis =1 )
X_test_selected_forest_with_names.to_excel("test_with_features_forest.xlsx")
X_train_selected_forest.to_excel("train_with_features_forest.xlsx")

#taking the data with the selcted features xgb model
selector_xgb = ForwardFeatureSelector_random_forest(n_features=5, cv=5)
X_train_selected_xgb = selector_xgb.fit_transform(df_standardized, y_train3)
X_test_selected_xgb = selector_xgb.transform(df_standardized_test.iloc[:,1:])
X_test_selected_xgb_with_names = pd.concat([df_var_test,X_test_selected_xgb],axis =1  )
X_test_selected_xgb_with_names.to_excel("test_with_features_xgb.xlsx")
X_train_selected_xgb.to_excel("train_with_features_xgb.xlsx")


#taking the data with the selcted features random forest model

