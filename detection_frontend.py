from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput

import numpy as np
from numpy.core.fromnumeric import size  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import itertools
import xgboost as xgb
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

# Base de dados
df = pd.read_csv('phishing_site_urls.csv', error_bad_lines=False)
# print(df.shape)

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
count_of_https_www = []  # quantidade de 'https'
count_of_www_dot = []  # cquantidade de 'www.'
count_of_not_alphanumeric = []  # quantidade de caracteres nao alfanumericos
count_of_hifen = []

for item in df['URL']:
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
        count_of_https_www.append(item.count("https://www"))
    except:
        count_of_https_www.append(0)

    try:
        count_of_www_dot.append(item.count("www."))
    except:
        count_of_www_dot.append(0)

    try:
        count_of_hifen.append(item.count("-"))
    except:
        count_of_hifen.append(0)

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
df['count_of_https_www'] = count_of_https_www
df['count_of_www_dot'] = count_of_www_dot
df['count_of_not_alphanumeric'] = count_of_not_alphanumeric
df['count_of_percentage'] = count_of_percentage
df['count_of_hifen'] = count_of_hifen

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


# Define X and y (entrada e saida)
X = df[['length_of_url', 'number_of_letters',
        'count_of_dotcom', 'count_of_codot', 'count_of_dotnet',
        'count_of_forward_slash', 'count_of_upper_case', 'count_of_lower_case',
        'count_of_dot', 'count_of_dot_info', 'count_of_https',
        'count_of_www_dot', 'count_of_not_alphanumeric', 'count_of_percentage', 'forwards_slash_to_length_ratio', 'count_of_hifen'
        ]]

y = df['Label']

# Divisao do dataset para treinamento e teste (90% e 10%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.10,
    random_state=234)

#print('Training dataset shape:', X_train.shape, y_train.shape)
#print('Testing dataset shape:', X_test.shape, y_test.shape)


# ------------------- CLASSIFICADOR ARVORE DE DECISÃO --------------------------

depth_list = []
accuracy_list = []

for depth in range(1, len(X.columns)):
    decision_tree = DecisionTreeClassifier(
        max_depth=50, splitter='best', criterion='gini', min_samples_split=2, min_samples_leaf=1)
    decision_tree.fit(X_train, y_train)
    accuracy = decision_tree.score(X_test, y_test)
    print('Depth: ', depth, ' Accuracy: ', accuracy)
    accuracy_list.append(decision_tree.score(X_test, y_test))
    depth_list.append(depth)


class DetectorURL(App):
    def build(self):
        self.window = GridLayout()
        self.window.cols = 1
        self.window.size_hint = (0.6, 0.7)
        self.window.pos_hint = {"center_x": 0.5, "center_y": 0.5}

        # add widgets to window

        # image widget
        self.window.add_widget(Image(source="logo.png", size_hint=(3, 3)))

        # Label widget
        self.url = Label(
            text="Insira a URL",
            font_size=20,
            color='#06ff80'
        )
        self.window.add_widget(self.url)

        # URL input widget
        self.input = TextInput(
            multiline=False,
            padding_y=(10, 10),
            size_hint=(1, 0.5)
        )
        self.window.add_widget(self.input)

        # button widget
        self.button = Button(
            text="DETECTAR",
            size_hint=(1, 0.5),
            bold=True,
            background_color='#06ff80'
        )
        self.button.bind(on_press=self.callback)
        self.window.add_widget(self.button)

        self.button = Button(
            text="RESETAR",
            size_hint=(1, 0.5),
            bold=True,
            background_color='#F44336',
            on_press=self.limpar
        )

        self.window.add_widget(self.button)

        return self.window

    def limpar(self, instance):
        self.url.text = "Insira a URL"

    def callback(self, instance):

        # Base de dados feita para teste
        df_teste = pd.read_csv('testes-url.csv', error_bad_lines=False)
        df_teste = df_teste.append({'url': self.input.text}, ignore_index=True)

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
        count_of_hifen = []

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
                count_of_hifen.append(item.count("-"))
            except:
                count_of_hifen.append(0)

            try:
                count_of_not_alphanumeric.append(
                    sum(not c.isalnum() for c in item))
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
        df_teste['count_of_hifen'] = count_of_hifen

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

        X_teste = df_teste[['length_of_url', 'number_of_letters',
                            'count_of_dotcom', 'count_of_codot', 'count_of_dotnet',
                            'count_of_forward_slash', 'count_of_upper_case', 'count_of_lower_case',
                            'count_of_dot', 'count_of_dot_info', 'count_of_https',
                            'count_of_www_dot', 'count_of_not_alphanumeric', 'count_of_percentage', 'forwards_slash_to_length_ratio', 'count_of_hifen'
                            ]]

        predictions = decision_tree.predict(X_teste)

        self.url.text = "A URL " + self.input.text + \
            " é considerada: " + predictions[22]


if __name__ == "__main__":
    DetectorURL().run()
