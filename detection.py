import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import itertools
import xgboost as xgb
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

# Base de dados
df = pd.read_csv('malicious_phish.csv', error_bad_lines=False)
print(df.shape)

length_of_url = []  # tamanho da URL
number_of_letters = []  # numero de letras
number_of_digits = []  # numero de digitos
count_of_dotcom = []  # quantidade de '.com'
count_of_codot = []  # quantidade de '.co.'
count_of_dotnet = []  # quantidade de '.net'
count_of_forward_slash = []  # quantidade de '/'
count_of_percentage = []  # quantidade de '%'
count_of_upper_case = []  # quantidade de caracteres em maiusculo
count_of_lower_case = []  # quantidade de caracteres em minusculo
count_of_dot = []  # quantidade de "."
count_of_upper_case = []  # quantidade de caracteres em maiusculo
count_of_lower_case = []  # quantidade de caracteres em minusculo
count_of_dot_info = []  # quantidade de '.info'
count_of_https = []  # quantidade de 'https'
count_of_www_dot = []  # cquantidade de 'www.'
count_of_not_alphanumeric = []  # quantidade de caracteres nao alfanumericos

for item in df['url']:
    try:
        length_of_url.append(len(item))
    except:
        length_of_url.append(0)

    try:
        number_of_letters.append(sum(c.isalpha() for c in item))
    except:
        number_of_letters.append(0)

    try:
        number_of_digits.append(sum(c.isdigit() for c in item))
    except:
        number_of_digits.append(0)

    try:
        count_of_dotcom.append(item.count(".com"))
    except:
        count_of_dotcom.append(0)

    try:
        count_of_codot.append(item.count(".co."))
    except:
        count_of_codot.append(0)

    try:
        count_of_dotnet.append(item.count(".net"))
    except:
        count_of_dotnet.append(0)

    try:
        count_of_forward_slash.append(item.count("/"))
    except:
        count_of_forward_slash.append(0)

    try:
        count_of_percentage.append(item.count("%"))
    except:
        count_of_percentage.append(0)

    try:
        count_of_dot.append(item.count("."))
    except:
        count_of_dot.append(0)

    try:
        count_of_upper_case.append(sum(c.isupper() for c in item))
    except:
        count_of_upper_case.append(0)

    try:
        count_of_lower_case.append(sum(c.islower() for c in item))
    except:
        count_of_lower_case.append(0)

    try:
        count_of_dot_info.append(item.count(".info"))
    except:
        count_of_dot_info.append(0)

    try:
        count_of_https.append(item.count("https"))
    except:
        count_of_https.append(0)

    try:
        count_of_www_dot.append(item.count("www."))
    except:
        count_of_www_dot.append(0)

    try:
        count_of_not_alphanumeric.append(sum(not c.isalnum() for c in item))
    except:
        count_of_not_alphanumeric.append(0)

df['length_of_url'] = length_of_url
df['number_of_letters'] = number_of_letters
df['number_of_digits'] = number_of_digits
df['count_of_dotcom'] = count_of_dotcom
df['count_of_codot'] = count_of_codot
df['count_of_dotnet'] = count_of_dotnet
df['count_of_forward_slash'] = count_of_forward_slash
df['count_of_upper_case'] = count_of_upper_case
df['count_of_lower_case'] = count_of_lower_case
df['count_of_dot'] = count_of_dot
df['count_of_upper_case'] = count_of_upper_case
df['count_of_lower_case'] = count_of_lower_case
df['count_of_dot_info'] = count_of_dot_info
df['count_of_https'] = count_of_https
df['count_of_www_dot'] = count_of_www_dot
df['count_of_not_alphanumeric'] = count_of_not_alphanumeric
df['count_of_percentage'] = count_of_percentage

# Amount of symbols to letters ratio
df['not_alphanumeric_to_letters_ratio'] = df['count_of_not_alphanumeric'] / \
    df['number_of_letters']

# Amount of '%' to length ratio
df['percentage_to_length_ratio'] = df['count_of_percentage']/df['length_of_url']

# Amount of '%' to length ratio
df['percentage_to_length_ratio'] = df['count_of_percentage']/df['length_of_url']

# Amount of '/' to length ratio
df['forwards_slash_to_length_ratio'] = df['count_of_forward_slash']/df['length_of_url']

# Amount captialised vs. non-capitalised
df['upper_case_to_lower_case_ratio'] = df['count_of_upper_case'] / \
    df['count_of_lower_case']

print(df.head())


# Define X and y
X = df[['length_of_url', 'number_of_letters', 'number_of_digits',
       'count_of_dotcom', 'count_of_codot', 'count_of_dotnet',
        'count_of_forward_slash', 'count_of_upper_case', 'count_of_lower_case',
        'count_of_dot', 'count_of_dot_info', 'count_of_https',
        'count_of_www_dot', 'count_of_not_alphanumeric', 'count_of_percentage',
        'percentage_to_length_ratio', 'forwards_slash_to_length_ratio']]

y = df['type']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=234)

print('Training dataset shape:', X_train.shape, y_train.shape)
print('Testing dataset shape:', X_test.shape, y_test.shape)

# ------------------- DECISION TREE CLASSIFIER --------------------------

depth_list = []
accuracy_list = []

for depth in range(1, len(X.columns)):
    decision_tree = DecisionTreeClassifier(max_depth=depth)
    decision_tree.fit(X_train, y_train)
    accuracy = decision_tree.score(X_test, y_test)
    print('Depth: ', depth, ' Accuracy: ', accuracy)
    accuracy_list.append(decision_tree.score(X_test, y_test))
    depth_list.append(depth)


# ------------- TESTANDO COM OUTRA BASE DE TESTE ------------------------

# Base de dados feita para teste
df_teste = pd.read_csv('testes-url.csv', error_bad_lines=False)
length_of_url = []  # tamanho da URL
number_of_letters = []  # numero de letras
number_of_digits = []  # numero de digitos
count_of_dotcom = []  # quantidade de '.com'
count_of_codot = []  # quantidade de '.co.'
count_of_dotnet = []  # quantidade de '.net'
count_of_forward_slash = []  # quantidade de '/'
count_of_percentage = []  # quantidade de '%'
count_of_upper_case = []  # quantidade de caracteres em maiusculo
count_of_lower_case = []  # quantidade de caracteres em minusculo
count_of_dot = []  # quantidade de "."
count_of_upper_case = []  # quantidade de caracteres em maiusculo
count_of_lower_case = []  # quantidade de caracteres em minusculo
count_of_dot_info = []  # quantidade de '.info'
count_of_https = []  # quantidade de 'https'
count_of_www_dot = []  # cquantidade de 'www.'
count_of_not_alphanumeric = []  # quantidade de caracteres nao alfanumericos

for item in df_teste['url']:
    try:
        length_of_url.append(len(item))
    except:
        length_of_url.append(0)

    try:
        number_of_letters.append(sum(c.isalpha() for c in item))
    except:
        number_of_letters.append(0)

    try:
        number_of_digits.append(sum(c.isdigit() for c in item))
    except:
        number_of_digits.append(0)

    try:
        count_of_dotcom.append(item.count(".com"))
    except:
        count_of_dotcom.append(0)

    try:
        count_of_codot.append(item.count(".co."))
    except:
        count_of_codot.append(0)

    try:
        count_of_dotnet.append(item.count(".net"))
    except:
        count_of_dotnet.append(0)

    try:
        count_of_forward_slash.append(item.count("/"))
    except:
        count_of_forward_slash.append(0)

    try:
        count_of_percentage.append(item.count("%"))
    except:
        count_of_percentage.append(0)

    try:
        count_of_dot.append(item.count("."))
    except:
        count_of_dot.append(0)

    try:
        count_of_upper_case.append(sum(c.isupper() for c in item))
    except:
        count_of_upper_case.append(0)

    try:
        count_of_lower_case.append(sum(c.islower() for c in item))
    except:
        count_of_lower_case.append(0)

    try:
        count_of_dot_info.append(item.count(".info"))
    except:
        count_of_dot_info.append(0)

    try:
        count_of_https.append(item.count("https"))
    except:
        count_of_https.append(0)

    try:
        count_of_www_dot.append(item.count("www."))
    except:
        count_of_www_dot.append(0)

    try:
        count_of_not_alphanumeric.append(sum(not c.isalnum() for c in item))
    except:
        count_of_not_alphanumeric.append(0)

df_teste['length_of_url'] = length_of_url
df_teste['number_of_letters'] = number_of_letters
df_teste['number_of_digits'] = number_of_digits
df_teste['count_of_dotcom'] = count_of_dotcom
df_teste['count_of_codot'] = count_of_codot
df_teste['count_of_dotnet'] = count_of_dotnet
df_teste['count_of_forward_slash'] = count_of_forward_slash
df_teste['count_of_upper_case'] = count_of_upper_case
df_teste['count_of_lower_case'] = count_of_lower_case
df_teste['count_of_dot'] = count_of_dot
df_teste['count_of_upper_case'] = count_of_upper_case
df_teste['count_of_lower_case'] = count_of_lower_case
df_teste['count_of_dot_info'] = count_of_dot_info
df_teste['count_of_https'] = count_of_https
df_teste['count_of_www_dot'] = count_of_www_dot
df_teste['count_of_not_alphanumeric'] = count_of_not_alphanumeric
df_teste['count_of_percentage'] = count_of_percentage

# Amount of symbols to letters ratio
df_teste['not_alphanumeric_to_letters_ratio'] = df_teste['count_of_not_alphanumeric'] / \
    df_teste['number_of_letters']

# Amount of '%' to length ratio
df_teste['percentage_to_length_ratio'] = df_teste['count_of_percentage'] / \
    df_teste['length_of_url']

# Amount of '%' to length ratio
df_teste['percentage_to_length_ratio'] = df_teste['count_of_percentage'] / \
    df_teste['length_of_url']

# Amount of '/' to length ratio
df_teste['forwards_slash_to_length_ratio'] = df_teste['count_of_forward_slash'] / \
    df_teste['length_of_url']

# Amount captialised vs. non-capitalised
df_teste['upper_case_to_lower_case_ratio'] = df_teste['count_of_upper_case'] / \
    df_teste['count_of_lower_case']

X_teste = df_teste[['length_of_url', 'number_of_letters', 'number_of_digits',
                    'count_of_dotcom', 'count_of_codot', 'count_of_dotnet',
                    'count_of_forward_slash', 'count_of_upper_case', 'count_of_lower_case',
                    'count_of_dot', 'count_of_dot_info', 'count_of_https',
                    'count_of_www_dot', 'count_of_not_alphanumeric', 'count_of_percentage',
                    'percentage_to_length_ratio', 'forwards_slash_to_length_ratio']]

predictions = decision_tree.predict(X_teste)
print(df_teste.shape)
print(df_teste.head)

# coeficientes de correlação com pishing
matcor = df.corr()
pd.set_option('display.max_rows', 17)
cor = abs(matcor['length_of_url'])
cor.sort_values(ascending=False)

print(predictions[0])
print(predictions[1])
print(predictions[2])
print(predictions[3])
print(predictions[4])
print(predictions[5])
print(predictions[6])
print(predictions[7])
print(predictions[8])
print(predictions[9])
print(predictions[10])
print(predictions[11])
