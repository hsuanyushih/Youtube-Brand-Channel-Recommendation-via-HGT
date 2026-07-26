# YouTube Brand–Creator Recommendation Pipeline

Scrapes YouTube data, detects brand mentions, builds a graph of YouTubers,
videos, and brands, trains a Graph Neural Network on it, and recommends
which creators a brand should collaborate with.

## Pipeline

1. **`01_data_scrape_ipynb.ipynb`** — Scrapes YouTube videos and channels via the YouTube Data API.
2. **`brand_extractor.ipynb`** — Uses the OpenAI API to detect brand names mentioned in each video.
3. **`02_HGT_heterodata_ipynb.ipynb`** — Builds a heterogeneous graph (youtuber / video / brand nodes and edges) from the labeled data.
4. **`03_HGT_final_training_ipynb.ipynb`** — Trains a Heterogeneous Graph Transformer (HGT) on the graph to predict brand–creator collaborations.
5. **`04_HGT_topk_recommendation.ipynb`** — Uses the trained model to recommend the best-fit creators for a given brand.

Run the notebooks in order — each one uses the file(s) produced by the previous step.

## Setup

```bash
pip install google-api-python-client httplib2 openai pandas numpy \
            scikit-learn matplotlib networkx tqdm jupyter ipywidgets \
            torch torch-geometric
```

(Install `torch` / `torch-geometric` following the
[official guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
for your hardware.)

Before running:
- Add your own YouTube API key in `01_data_scrape_ipynb.ipynb`
- Add your own OpenAI API key in `brand_extractor.ipynb`
- Each notebook has placeholder file names (e.g. `YOUR_INPUT_FILE.json`,
  `YOUR_DATASET_FILE.json`, `YOUR_GRAPH_OUTPUT_FILE.pt`) near the top —
  replace these with your actual file names. The same placeholder name
  must be used consistently between the notebook that creates a file and
  the notebook(s) that read it.
