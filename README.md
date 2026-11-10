# Sentiment Analysis of Amazon Fine Food Reviews with Deep Learning

Three-class sentiment classification (**negative / neutral / positive**, labels derived from the star rating) of the text of a food review, with two neural networks:
a **BiLSTM with attention** trained from scratch and a **BERT-base** model fine-tuned on the data.
The reference is the rule "always answer positive". The main metric is Macro-F1.

## Results (full test set, 39,358 reviews)

| Model | Accuracy | Macro-F1 | Parameters |
|---|---|---|---|
| BERT-base, fine-tuned | 0.9074 | **0.7877** | 109.48 M |
| BiLSTM + attention | 0.8788 | 0.7422 | 6.25 M |
| Always positive | 0.7794 | 0.2920 | - |

BERT is ahead of the BiLSTM by 0.0454 Macro-F1 (95% bootstrap interval [+0.0400, +0.0507]). Neutral is the hardest class for both models.
Details are in notebook 06 and in the reports.

## Dataset

[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) (Stanford Network Analysis Project, Kaggle): 568,454 food reviews posted on Amazon between October 1999 and October 2012, each with a 1-5 star rating.
Source paper: J. McAuley and J. Leskovec, *From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews*, WWW 2013.

The data are not stored in this repository. Download `Reviews.csv` from the Kaggle page above and place it at `dataset/Reviews.csv` (create the folder) before running notebook 01.

## Repository layout

```
notebooks/
  01_data_cleaning.ipynb              problem summary, data-quality checks, removal of repeated reviews
  02_data_understanding.ipynb         exploratory data analysis and the design decisions it implies
  03_data_preparation.ipynb           model input text, stratified split, class weights
  04_model_bilstm_attention.ipynb     model 1: BiLSTM + attention
  05_model_bert.ipynb                 model 2: BERT-base fine-tuning
  06_evaluation.ipynb                 results, bootstrap intervals, threshold calibration, error analysis
  07_demo.ipynb                       load the saved models and try your own sentences
  sentiment_analysis_colab.ipynb      all-in-one executed notebook (Vietnamese, with outputs)
app.py                              Streamlit web demo
src/                                shared code
  config.py                           paths, labels, hyperparameters (CFG), seed, device, QUICK_RUN
  text.py                             text cleaning, tokenisation
  data.py                             intermediate data I/O, vocabulary, Dataset, length-bucketing sampler
  models.py                           BiLSTMAttention
  training.py                         BiLSTM training loop (fit_bilstm) and inference
  transformer.py                      HuggingFace inference and fine-tuning (fit_transformer)
  evaluation.py                       metrics, confusion matrices, bootstrap, neutral-threshold calibration, saving/loading predictions
  explain.py                          attention visualisation
outputs/                            generated files: split data (data/), saved models (bilstm/, bert/), predictions (predictions/)
report/                             LaTeX report, English (main.tex, content/)
report_vi/                          LaTeX report, Vietnamese
slides/                             Beamer slides, HUST red theme (slides.tex)
```

## Not included in the repository

GitHub rejects files over 100 MB, so these are excluded (see `.gitignore`) and must be recreated:

- `dataset/`: the raw data, see the Dataset section above.
- `outputs/data/*.pkl`: created by notebooks 01 and 03.
- Model weights `outputs/bilstm/model.pt` and `outputs/bert/model.safetensors`: created by notebooks 04 and 05 (BERT needs a GPU for a reasonable training time). The web demo needs them.

Predictions, training histories and the results table in `outputs/` are included.

## Running the notebooks

Open the notebooks from the `notebooks/` folder (Jupyter or VS Code) and run them in order, 01 to 06 (07 only needs the models saved by 04 and 05). The first code cell adds the project folder to the Python path, and each notebook reads the output of the previous one from `outputs/`.

```
pip install pandas numpy matplotlib seaborn nltk scikit-learn tqdm ipywidgets torch transformers
```

- `QUICK_RUN=1` (environment variable, or `QUICK_RUN` in `src/config.py`) runs the whole chain on 30,000 reviews to check the code. **Note:** it overwrites `outputs/data/`.
- Notebooks 04 and 05 have a `RETRAIN` flag. With `False` (default) they load the saved model from `outputs/` if it exists instead of training again.
- On a CPU, BERT is fine-tuned on a 20,000-review subset with `max_len = 128`; on a GPU it uses the whole training set with `max_len = 256`. CPU results are not comparable with the table above.
- The notebooks are saved without outputs. Run them to regenerate tables and figures (a GPU is recommended for notebooks 05 and 06).
- `notebooks/sentiment_analysis_colab.ipynb` is the same pipeline in one file, already executed. It writes its own copy of `src/` into the project folder when run.

## Web demo (Streamlit)

Type an English review and the app returns the predicted label (negative / neutral / positive) and the class probabilities of each model, together with the words the BiLSTM attends to most.
A second tab scores many reviews at once (one per line) and exports the result as CSV.

```
pip install streamlit torch transformers numpy pandas scikit-learn tqdm ipython
streamlit run app.py
```

It needs `outputs/bilstm/` (`model.pt`, `vocab.json`) and `outputs/bert/` (model and tokenizer), created by notebooks 04 and 05. It runs on a CPU; the first start takes a few seconds to load the models.

## Reports and slides

```
cd report      && latexmk -xelatex main.tex          # English report
cd report_vi   && latexmk -xelatex main.tex          # Vietnamese report
cd slides      && latexmk -lualatex slides.tex       # slides (needs LuaLaTeX)
```

## Notes

- Labels are weak: they come from star ratings, so a 5-star review can still contain a complaint.
- The data are English food reviews from Amazon (1999-2012); other domains may work worse.
- The split is random, not chronological, so scores do not measure performance on future reviews.
