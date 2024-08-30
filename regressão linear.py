import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Gerar dados simulados para o cenário
np.random.seed(42)
quantidade_dados = np.random.randint(1, 50, 50)  # até 50 valores para quantidade de dados
tempo_sumarizacao = 1.5 * quantidade_dados + np.random.normal(0, 5, quantidade_dados.size)  # tempo com variabilidade

# Criar modelo de regressão linear
modelo_sumarizacao = LinearRegression()

# Ajustar os dados para o formato esperado pelo modelo (reshape)
quantidade_dados_reshaped = quantidade_dados.reshape(-1, 1)

# Treinar o modelo
modelo_sumarizacao.fit(quantidade_dados_reshaped, tempo_sumarizacao)

# Fazer previsões para o mesmo conjunto de dados
tempo_previsto = modelo_sumarizacao.predict(quantidade_dados_reshaped)

# Avaliar a qualidade do modelo
mse_sumarizacao = mean_squared_error(tempo_sumarizacao, tempo_previsto)
r2_sumarizacao = r2_score(tempo_sumarizacao, tempo_previsto)

# Exibir os dados sumarizados (observados e previstos)
print("Quantidade de Dados | Tempo Observado (s) | Tempo Previsto (s)")
for q, t_obs, t_prev in zip(quantidade_dados, tempo_sumarizacao, tempo_previsto):
    print(f"{q:<18} | {t_obs:<18.2f} | {t_prev:<18.2f}")

# Plotar gráfico de dispersão e a linha de regressão
plt.scatter(quantidade_dados, tempo_sumarizacao, color='blue', label='Dados Observados')
plt.plot(quantidade_dados, tempo_previsto, color='red', linewidth=2, label='Linha de Regressão')
plt.xlabel('Quantidade de Dados Enviados')
plt.ylabel('Tempo de Sumarização (s)')
plt.title('Relação entre Quantidade de Dados e Tempo de Sumarização')
plt.legend()
plt.grid(True)
plt.show()

# Exibir os resultados de MSE e R²
print(f"\nMSE: {mse_sumarizacao:.2f}")
print(f"R²: {r2_sumarizacao:.2f}")