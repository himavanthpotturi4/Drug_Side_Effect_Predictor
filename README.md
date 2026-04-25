# Drug Side-Effect Prediction System

A Streamlit-based drug side-effect prediction website powered by a graph neural network (GNN). The project builds a drug knowledge graph, trains an architecture-aligned model, and serves fast predictions for drug side effects, primary target type, and common uses.

## Features

- Predict likely side effects for a drug
- Predict a primary biological target type
- Show common uses for supported drugs
- Train the full pipeline from data collection to model export
- Save model-level evaluation artifacts in `outputs/`
- Keep evaluation outputs out of the website UI

## Website

The website is built with Streamlit and currently shows:

- drug search input
- random drug button
- likely side effects
- primary target type
- common uses

The website does **not** display:

- confusion matrices
- embedding visualizations
- evaluation metrics
- knowledge graph sample visualizations
- side-effect probability numbers

Those are still generated during training and saved into `outputs/`.

## Project Structure

```text
app.py
scripts/
  download_data.py
  train_model.py
  run_architecture_pipeline.py
src/drug_predictor/
  architecture/
  config.py
  data_collection.py
  inference.py
  model.py
  preprocessing.py
  train.py
data/
  raw/
  processed/
models/
outputs/
```

## Requirements

- Python 3.10+
- Windows/macOS/Linux

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The Website

Start the Streamlit app:

```bash
streamlit run app.py
```

Default local URL:

```text
http://127.0.0.1:8501
```

## Train The Full Architecture Pipeline

This runs:

1. data collection
2. preprocessing
3. knowledge graph construction
4. model training
5. evaluation artifact generation

Command:

```bash
python scripts/run_architecture_pipeline.py --max-drugs 300 --epochs 50
```

Arguments:

- `--max-drugs`: limits how many drugs are collected/processed
- `--epochs`: number of training epochs

## Other Scripts

Download data and build base training tables:

```bash
python scripts/download_data.py --max-drugs 1200
```

Train the non-architecture training path:

```bash
python scripts/train_model.py
```

## Generated Outputs

After training, model-level outputs are written to `outputs/`.

### Evaluation Metrics

- `architecture_test_metrics.json`
- `architecture_test_metrics.png`
- `architecture_training_history.csv`

### Confusion Matrices

- `confusion_matrix_action.csv`
- `confusion_matrix_action.png`
- `confusion_matrix_side_effect_binary.csv`
- `confusion_matrix_side_effect_binary.png`
- `confusion_matrix_side_effect_binary_summary.json`

### Embedding Visualization

- `embedding_projection_drugs.csv`
- `embedding_projection_drugs.png`

### Knowledge Graph Visualization

- `knowledge_graph_sample.png`
- `knowledge_graph_sample_presentation.png`
- `knowledge_graph_sample_summary.json`

## Model Outputs vs Website Outputs

### Website outputs

These change when the user enters a different drug:

- predicted side effects
- predicted target type
- common uses

### Model-level outputs

These stay the same until the model is retrained:

- evaluation metrics
- confusion matrices
- embedding projection
- knowledge graph sample images
- saved model weights

## Performance Notes

The app has been optimized to improve responsiveness:

- predictor is cached with `st.cache_resource`
- graph visualization is not regenerated during website inference
- model logits are precomputed once and reused
- prediction form avoids rerunning the app on every keystroke
- common uses use a fast local fallback/cache

This means:

- initial app load is faster
- typing into the input field feels smoother
- prediction response is much faster after the model is loaded

## Common Uses

The website shows common uses from:

- local processed/cache data when available
- a curated local fallback list for common drugs

If a drug has no supported local use data, the app shows:

```text
No treatment indication data found for this drug in the current dataset.
```

## Notes

- The current model quality is better for side-effect prediction than for action classification.
- Evaluation metrics are computed across the full test set, not for a single drug.
- Searching a different drug changes the prediction result, but not the saved model evaluation artifacts.

## Troubleshooting

### Website says model is not ready

Run:

```bash
python scripts/run_architecture_pipeline.py --max-drugs 300 --epochs 50
```

### Website changes are not showing up

Restart Streamlit:

```bash
streamlit run app.py
```

### Common Uses is empty for a known drug

Restart the app after code changes so the cached predictor reloads.

## License

Add your preferred license here.
