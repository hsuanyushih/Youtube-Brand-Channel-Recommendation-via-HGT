# YouTube Brand–Creator Recommendation via Heterogeneous Graph Transformer (HGT)

An end-to-end recommendation system that leverages **Large Language Models (OpenAI GPT)** for semantic brand entity extraction and **Heterogeneous Graph Transformers (HGTs)** for brand–creator link prediction on YouTube.

The pipeline automatically collects YouTube videos and channel metadata through the YouTube Data API, uses LLMs to identify real-world brand mentions from unstructured video content, constructs a heterogeneous graph of creators, videos, and brands, and generates **Top-k creator recommendations** for potential brand collaborations.

This repository includes the complete machine learning pipeline, covering **data collection, LLM-assisted annotation, heterogeneous graph construction, HGT model training, and recommendation generation**.

> **Thesis:** *YouTube Brand–Creator Collaboration Recommendation via Heterogeneous Graph Neural Networks*
## Project Structure

```
.
├── 01_data_scrape_ipynb.ipynb           # Scrape raw video/channel data via YouTube API
├── 02_brand_extractor.ipynb             # Detect brand mentions via the OpenAI API
├── 03_HGT_heterodata_ipynb.ipynb        # Data construction & feature cleaning
├── 04_HGT_final_training_ipynb.ipynb    # 5-fold CV training
├── 05_HGT_topk_recommendation.ipynb     # Top-k recommendation
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/HGT-YouTube-Brand-Recommendation.git
cd HGT-YouTube-Brand-Recommendation
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

pip install google-api-python-client httplib2 openai pandas numpy \
            scikit-learn matplotlib networkx tqdm jupyter ipywidgets \
            torch torch-geometric
```

> `torch` and `torch-geometric` installation commands vary by OS and CUDA version. Please follow the official installation guides instead of a generic `pip install`:
> - [PyTorch installation guide](https://pytorch.org/get-started/locally/)
> - [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

## Reproducing the Experiments

Run the notebooks in order. Each notebook has placeholder file names near the top (e.g. `YOUR_INPUT_FILE.json`, `YOUR_DATASET_FILE.json`, `YOUR_GRAPH_OUTPUT_FILE.pt`) — replace these with your actual file names, keeping the same name consistent between the notebook that creates a file and the notebook(s) that read it.

1. **`01_data_scrape_ipynb.ipynb`**
   Scrapes video and channel data via the YouTube Data API. Requires your own YouTube API key. Outputs a raw dataset JSON.

2. **`02_brand_extractor.ipynb`**
   Uses the OpenAI API to detect real-world brand names mentioned in each video's title/description. Requires your own OpenAI API key.

3. **`03_HGT_heterodata_ipynb.ipynb`**
   Loads the labeled dataset, performs brand feature cleaning, and builds a heterogeneous graph (youtuber / video / brand nodes and edges) as a PyTorch Geometric `HeteroData` object.

4. **`04_HGT_final_training_ipynb.ipynb`**
   Trains the HGT model with 5-fold cross-validation, using validation AUC for early stopping and validation-loss protection to prevent overfitting. Saves the best checkpoint per fold, along with fold summaries, learning curves, and brand embeddings.

5. **`05_HGT_topk_recommendation.ipynb`**
   Loads the graph data and the trained fold checkpoints, generates Top-k brand-channel collaboration recommendations using a 5-fold ensemble average score, and excludes existing collaborations.

## Results

Example: Samsung's Top-5 collaboration recommendations are demonstrated in `05_HGT_topk_recommendation.ipynb`.

## Citation

This repository accompanies the thesis:

> *YouTube Brand-Channel Collaboration Recommendation via Heterogeneous Graph Neural Networks*

```bibtex
@mastersthesis{Shih, Hsuan-Yu_2026,
  title  = {YouTube Brand-Channel Collaboration Recommendation via Heterogeneous Graph Neural Networks},
  author = {Shih, Hsuan-Yu},
  school = {National Taipei University},
  year   = {2026}
}
```

(The thesis itself is not included in this repository. Citation details will be updated once it is publicly available.)
