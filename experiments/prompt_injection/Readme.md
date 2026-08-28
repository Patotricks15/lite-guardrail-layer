# Prompt Injection Experiment

Experimento reproduzivel do detector de prompt injection baseado exclusivamente no `user_prompt`.

## Dados

O notebook consome `data/prompt_injection.parquet`, que combina SPML, `deepset/prompt-injections` e os labels `prompt-injection`/`safe` de `Yash0728/toxicity_prompt-injection`. O split e agrupado por `user_prompt` para evitar leakage.

## Features

- embedding MiniLM do user prompt: 384 dimensoes
- nove flags lexicas de injection
- total: 393 features

O `system_prompt` nao e usado por este modelo.

## Tuning, calibracao e threshold

O notebook compara Decision Tree, Random Forest e XGBoost para Baseline e Baseline + LDA. Cada algoritmo testa uma grade compacta de tres configuracoes dos principais hiperparametros. Para cada configuracao, compara calibracao `sigmoid` e `isotonic` em uma particao reservada da validacao. O threshold e escolhido com recall-alvo de 99,5%. O vencedor e selecionado pelo maior PR-AUC de validacao, sem consultar o teste.

## Latencia

O notebook mede o preprocessing em lote e o `predict_proba` depois de um aquecimento, usando a mediana de cinco repeticoes. `end_to_end_latency_ms_per_sample` e a soma desses componentes. Os valores representam milissegundos por amostra no batch de teste e no ambiente da execucao; nao representam p95 de requisicoes online.

O vencedor foi XGBoost / Baseline + LDA (`max_depth=6`, `learning_rate=0.1`, `n_estimators=300`). Com threshold experimental fixo em 0,2, obteve 98,54% de accuracy, 98,87% de precision, 99,26% de recall, 4,0% de FPR e PR-AUC de 99,91%. A latencia end-to-end foi 31,26 ms por amostra.

## Execucao

Abra `notebook.ipynb` com o ambiente `.venv` e execute todas as celulas. O notebook treina e calibra o modelo, avalia o split de teste e sobrescreve os resultados.

## Resultados

O diretorio `results/` contem:

- `confusion_matrix.png`
- `calibration_plot.png`
- `confusion_matrix_counts.png`
- `confusion_matrix_percentages.png`
- `threshold_analysis.png`
- `results.xlsx`

O workbook inclui a aba `hyperparameter_tuning`, comparacoes de validacao e teste, hiperparametros vencedores, calibracao, latencias, matrizes de confusao, analise de threshold e distribuicao do dataset.
