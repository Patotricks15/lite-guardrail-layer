# Out-of-Context Experiment

Experimento reproduzivel do detector que compara semanticamente `system_prompt` e `user_prompt`.

## Dados

O notebook consome `data/out-of-context.parquet`. Exemplos positivos sao mismatches sinteticos entre user prompts seguros e outros system prompts do mesmo split. O split e agrupado por `system_prompt`.

## Features

- user embedding completo: 384 dimensoes
- diferenca absoluta entre system e user por dimensao: 384 dimensoes
- cosine similarity
- distancia de Jaccard entre os conjuntos de tokens normalizados
- dot product

O classificador recebe 771 features. O embedding completo do `system_prompt` nao entra diretamente; ele participa da diferenca absoluta e das tres metricas relacionais.

## Tuning, calibracao e threshold

O notebook compara Decision Tree, Random Forest e XGBoost. Cada algoritmo testa uma grade compacta de tres configuracoes dos principais hiperparametros. Para cada configuracao, compara calibracao `sigmoid` e `isotonic` em uma particao reservada da validacao. O threshold e escolhido no ponto em que precision e recall sao iguais ou o mais proximos possivel. O vencedor e selecionado pelo maior PR-AUC de validacao, sem consultar o teste.

## Latencia

O notebook mede o preprocessing em lote e o `predict_proba` depois de um aquecimento, usando a mediana de cinco repeticoes. `end_to_end_latency_ms_per_sample` e a soma desses componentes. Os valores representam milissegundos por amostra no batch de teste e no ambiente da execucao; nao representam p95 de requisicoes online.

O vencedor foi XGBoost (`max_depth=4`, `learning_rate=0.05`, `n_estimators=400`), com 81,36% de recall, PR-AUC de 89,33% e latencia end-to-end de 5,97 ms por amostra.

## Execucao

Abra `notebook.ipynb` com o ambiente `.venv` e execute todas as celulas. O notebook treina e calibra o modelo, avalia o split de teste e sobrescreve os resultados.

### Retreino Com HF Off-Topic (10%)

Para o experimento com `https://huggingface.co/datasets/gabrielchua/off-topic`, use o script:

```bash
.venv/bin/python experiments/out_of_context/retrain_with_hf_10pct.py
```

O script:

- carrega `data/out-of-context.parquet` (dataset local)
- carrega `gabrielchua/off-topic` (HF)
- amostra 10% estratificado por `off_topic`
- normaliza colunas para `system_prompt`, `user_prompt`, `out_of_context`
- cria split `train/validation/test` no sample HF por grupo de `system_prompt`
- concatena local + HF em um unico parquet
- retreina o `out_of_context` usando esse parquet concatenado

Arquivos gerados em `experiments/out_of_context/results/`:

- `out-of-context-merged-10pct.parquet`
- `out_of_context_model_merged_10pct.pkl`
- `retrain_merged_10pct_summary.json`

## Resultados

O diretorio `results/` contem:

- `calibration_plot.png`
- `confusion_matrix_counts.png`
- `confusion_matrix_percentages.png`
- `threshold_analysis.png` (modelo vencedor)
- `results.xlsx`

O workbook inclui a aba `hyperparameter_tuning`, comparacoes de validacao e teste, hiperparametros vencedores, calibracao, latencias, analise de threshold e matrizes de confusao do vencedor.

## Limitacao

Os rotulos positivos sao sinteticos. System prompts diferentes podem ser semanticamente compativeis, introduzindo ruido e falsos positivos. Use dados reais anotados antes de uma decisao de producao.
