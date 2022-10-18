# IMPORTAR BASE DE DADOS
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor  # Arvore de Decisão
from sklearn.linear_model import LinearRegression  # RegressãoLinear
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
tabela = pd.read_csv("D:\Intensivão Python\AULA 4/advertising.csv")
display(tabela)

# TRATAMENTO DE DADOS
print(tabela.info())

# CRIAR O GRÁFICO DE CORRELAÇÃO.
#!pip install matplotlib
#!pip install seaborn
#!pip install scikit-learn

# CRIAR GRÁFICO
sns.heatmap(tabela.corr(), cmap="Wistia", annot=True)

# Exibir grafico
plt.show()


# SEPARAR DADOS DE TREINO E DADOS DE TESTE
# eixo X -> quem eu quero calcular
# eixo Y -> quem eu quero prever

y = tabela["Vendas"]  # quem eu quero prever
x = tabela[["TV", "Radio", "Jornal"]]  # quem eu quero usar para calcular

# separar os dados em treino e teste

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, git test_size=0.3)


# CRIAR A INTELIGÊNCIA ARTIFICIAL
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()


# TREINAR A INTELIGÊNCIA ARTIFICIAL
# Regressão Linear -> plota pontinhos no gráfico e traça um linha reta.
modelo_regressaolinear.fit(x_treino, y_treino)
# Árvore de decisõo -> faz perguntas para a base de dados
modelo_arvoredecisao.fit(x_treino, y_treino)


# Qual é o melhor modelo? (quem ganha o jogo)

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

# TESTE DE IA e ESCOLHA DO MELHOR MODELO.
# - Quanto mais perto de 1, melhor é a eficácia da IA
print(metrics.r2_score(y_teste, previsao_regressaolinear))
print(metrics.r2_score(y_teste, previsao_arvoredecisao))


tabela_auxiliar = pd.DataFrame()
tabela_auxiliar['y_teste'] = y_teste
tabela_auxiliar['Previsao arvore de Decisao'] = previsao_arvoredecisao
tabela_auxiliar['Previsao regressao linear'] = previsao_regressaolinear

plt.figure(figsize=(15, 6))
sns.lineplot(data=tabela_auxiliar)
plt.show()
