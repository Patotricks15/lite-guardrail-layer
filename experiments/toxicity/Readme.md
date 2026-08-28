# Toxicity Experiment

Experimento reproduzivel do detector de toxicity baseado exclusivamente no `user_prompt`.

## Dados

O notebook consome `data/toxicity.parquet`, derivado de `Yash0728/toxicity_prompt-injection`. O problema e binario: `safe` recebe classe 0 e `toxicity` recebe classe 1. Registros `prompt-injection` sao excluidos deste experimento e usados somente para enriquecer `data/prompt_injection.parquet`. O split e agrupado por `user_prompt` para evitar leakage.

## Features

- embedding MiniLM do user prompt: 384 dimensoes
- nove flags lexicas compartilhadas com o pipeline de prompt injection
- total: 393 features
- variante experimental com um score LDA cross-fitted adicional

## Tuning, calibracao e threshold

O notebook compara Decision Tree, Random Forest e XGBoost para Baseline e Baseline + LDA. Cada algoritmo testa uma grade compacta de tres configuracoes dos principais hiperparametros. Para cada configuracao, compara calibracao `sigmoid` e `isotonic` em uma particao reservada da validacao. O threshold e escolhido com recall-alvo de 99,5%. O vencedor e selecionado pelo maior PR-AUC de validacao, sem consultar o teste.

## Latencia

O notebook mede o preprocessing em lote e o `predict_proba` depois de um aquecimento, usando a mediana de cinco repeticoes. `end_to_end_latency_ms_per_sample` e a soma desses componentes. Os valores representam milissegundos por amostra no batch de teste e no ambiente da execucao; nao representam p95 de requisicoes online.

O vencedor foi Decision Tree / Baseline (`max_depth=12`, `min_samples_leaf=5`), com 96,43% de recall, PR-AUC de 96,21% e latencia end-to-end de 4,30 ms por amostra.

## Execucao

Execute `src/build_datasets.py` antes do notebook. Abra `notebook.ipynb` com o ambiente `.venv` e execute todas as celulas. O notebook treina e calibra os candidatos, avalia o split de teste e sobrescreve os resultados.

## Resultados

O diretorio `results/` contem:

- `calibration_plot.png`
- `confusion_matrix_counts.png`
- `confusion_matrix_percentages.png`
- `threshold_analysis.png`
- `results.xlsx`

O workbook inclui a aba `hyperparameter_tuning`, comparacoes de validacao e teste, hiperparametros vencedores, calibracao, latencias, analise de threshold, matrizes de confusao e distribuicao do dataset.
