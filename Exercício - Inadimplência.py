import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Função para separar treino e teste
# Métricas de avaliação do modelo programadas no scikit
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix, balanced_accuracy_score

# Classe de árvore e funções auxiliares
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split

#%% Definir uma semente aleatória para reprodutibilidade
np.random.seed(42)
random.seed(42)

# Gerar as variáveis simuladas com correlação
idade = np.random.randint(18, 71, 10000)

# Gerar variáveis correlacionadas usando a função multivariada normal
mean_values = [5000, 2000, 0.5, 5]  # Médias das variáveis
correlation_matrix = np.array([
    [1, 0.3, 0.2, -0.1],
    [0.3, 1, -0.1, 0.2],
    [0.2, -0.1, 1, 0.4],
    [-0.1, 0.2, 0.4, 1]
])  # Matriz de correlação

# Gerar dados simulados
simulated_data = np.random.multivariate_normal(mean_values, correlation_matrix, 10000)

renda = simulated_data[:, 0]
divida = simulated_data[:, 1]
utilizacao_credito = np.clip(simulated_data[:, 2], 0, 1)  # Limita a utilização de crédito entre 0 e 1
consultas_recentes = np.maximum(simulated_data[:, 3], 0)  # Garante que o número de consultas recentes seja não negativo

# Gerar função linear das variáveis explicativas
preditor_linear = -7 - 0.01 * idade - 0.0002 * renda + 0.003 * divida - 3 * utilizacao_credito + 0.5 * consultas_recentes

# Calcular probabilidade de default (PD) usando a função de link logit
prob_default = 1 / (1 + np.exp(-preditor_linear))

# Gerar inadimplência como variável Bernoulli com base na probabilidade de default
inadimplencia = np.random.binomial(1, prob_default, 10000)

# Criar dataframe
dados = pd.DataFrame({
    'idade': idade,
    'renda': renda,
    'divida': divida,
    'utilizacao_credito': utilizacao_credito,
    'consultas_recentes': consultas_recentes,
    'inadimplencia': inadimplencia
})

print(dados.head())

# Categorizar a idade
kbin_idade = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
dados['idade_cat'] = kbin_idade.fit_transform(dados[['idade']])

def descritiva2(var1, var2, df):
    cross_tab = pd.crosstab(df[var1], df[var2], normalize='index')
    print(cross_tab)

descritiva2('idade_cat', 'inadimplencia', dados)

print(dados.head())

dados.to_parquet('exercicio.parquet')

#%% Começa a resolução do exercicio

#Aqui achei importante verificar se na DF tem mais inadimplentes ou não.
contagem = dados['inadimplencia'].value_counts()
print(contagem)



#Tratando os dados
#Vamos criar categorias paras as variaveis "Consultas Recentes", "Renda", "Divida", "Utilização de Credito".
#Aqui Decidi criar um kbin para cada, caso futuramente eu escolha modificar em algum parametro especifico não afetar os demais.

kbin_consulta = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
dados['consultas_recentes_cat'] = kbin_consulta.fit_transform(dados[['consultas_recentes']])

kbin_renda = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
dados['renda_cat'] = kbin_renda.fit_transform(dados[['renda']])

kbin_divida = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
dados['divida_cat'] = kbin_divida.fit_transform(dados[['divida']])

kbin_credito = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
dados['utilizacao_credito_cat'] = kbin_credito.fit_transform(dados[['utilizacao_credito']])


#Excluindo colunas duplicadas:
dados.drop(columns= ['idade', 'divida', 'consultas_recentes', 'renda', 'utilizacao_credito'], inplace= True)

#%% Dividindo em conjunto de Treino e de Teste

#Para isso vamos usar train_teste-split() do sklerarn
#Definindo X e Y que vão entrar na formula.
X = dados[['idade_cat', 'consultas_recentes_cat','renda_cat', 'divida_cat', 'utilizacao_credito_cat']]
Y = dados['inadimplencia']

#Utilizando a formula
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

'''
A fomula tem a saida X_train, X_test, Y_train, Y_test, que é usado para treinar e testar o modelo.
o test_size=0,3, siguinifica que a divizão é de 30% para testar e 70% para treinar.
random_state=42 fixa uma semente aleatoria que garante que a mesma contidade de dados serão divididos sempre que rodar o codigo
'''


#%% Iniciando a arvore

#Vamos criar a arvore com a função DecisionTreeClassifier()
arvore = DecisionTreeClassifier(criterion= 'gini', max_depth= 3, random_state= 42, min_samples_split= 20, min_samples_leaf= 10)

#Vamos treinar a arvore com a função .fit()
arvore.fit(X_train , Y_train)

#Vamos vizualizar a arvore criada
plt.figure(figsize= (20, 10))
plot_tree(arvore, feature_names= X_train.columns.tolist(), class_names= ['não inadiplente', 'inadiplente'], filled= True)
plt.title("Árvore de Decisão - Classificação de Inadimplencia")
plt.show()


#%% Avaliando a Classificação

#Guardando a classificação
classificacao_treino = arvore.predict(X_test)
 
#Comparando os valores
print(pd.crosstab(classificacao_treino, Y_test, margins= True))
print(pd.crosstab(classificacao_treino, Y_test, normalize= 'index'))
print(pd.crosstab(classificacao_treino, Y_test, normalize= 'columns'))
 
acertos = classificacao_treino == Y_test
pct_acertos = acertos.sum()/acertos.shape[0]
print(f'Acuracia(taxa de acerto): {pct_acertos:.2%}')

#%% Calculando a acuracia e matriz de confusão

#Vamos usar o Scikit-Lern
#Para calcular a matrix de conusão, use confusion_matrix()
matrix_confusao = confusion_matrix(Y_test , classificacao_treino)

#Para calcular a acuracia, use accuracy_score
acuracia = accuracy_score(Y_test, classificacao_treino)


#Para calcular a acuracia balanceada, use accuracy_score()
acuracia_balanceada = balanced_accuracy_score(Y_test, classificacao_treino)

print(f'\nA acurácia da árvore é : {acuracia:.1%}')
print(f'A acurácia balanceada da árvore é: {acuracia_balanceada : .1%}')

#%% Vizualizar em grafico

sns.heatmap(matrix_confusao, 
            annot=True, fmt='d', cmap='viridis', 
            xticklabels=['Não inadimplete', 'inadimplete'], 
            yticklabels=['Não inadimplete', 'inadimplete'])
plt.title("Matriz de Confusão - Classificação de Inadimplência")
plt.show()

#Relatorio de classificação
print('\n', classification_report(Y_test, classificacao_treino))
