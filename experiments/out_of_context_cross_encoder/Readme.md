# Out-of-Context Cross-Encoder Experiment

Experimento zero-shot com `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` diretamente sobre os pares `system_prompt` e `user_prompt`.

## Modelo

O modelo e um MiniLM multilingue pre-treinado para relevancia e e usado sem fine-tuning. Scores maiores do checkpoint representam maior relevancia; o notebook inverte o score para que valores maiores representem maior evidencia de out-of-context.

- comprimento maximo combinado: 256 tokens
- batch de inferencia: 32
- nenhum peso do Transformer e atualizado

O mesmo `data/out-of-context.parquet` e os mesmos splits isolados por `system_prompt` e `pair_id` do experimento baseado em XGBoost sao usados.

## Calibracao e threshold

A validacao e dividida por `pair_id`. Uma metade ajusta calibradores leves sigmoid e isotonic e os compara pelo menor Brier score. A outra metade escolhe o threshold em que precision e recall sao iguais ou o mais proximos possivel. O teste nao participa dessas escolhas.

## Metricas

O notebook exporta:

- accuracy
- precision
- recall
- F1
- false positive rate e false negative rate
- falsos positivos e falsos negativos
- ROC-AUC e PR-AUC
- latencia de tokenizacao
- latencia do forward do Transformer
- latencia end-to-end
- matrizes de confusao em contagem e percentual

## Execucao

Abra `notebook.ipynb` com o ambiente `.venv` e execute todas as celulas em ordem. O ambiente atual nao possui GPU, portanto a inferencia do cross-encoder e as cinco repeticoes da medicao de latencia podem demorar em CPU, mas nao ha etapa de treinamento do Transformer.

## Resultados

A execucao preenche `results/` com:

- `calibration_plot.png`
- `confusion_matrix_counts.png`
- `confusion_matrix_percentages.png`
- `threshold_analysis.png`
- `results.xlsx`

O workbook possui as abas `dataset_distribution`, `model_configuration`, `calibration`, `validation_comparison`, `test_comparison`, `threshold_analysis`, `confusion_counts` e `confusion_percentages`.
