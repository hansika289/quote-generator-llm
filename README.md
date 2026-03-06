## Mini LLM Quote Generator

This project is a tiny character-level GPT-style language model that learns from a text file of quotes and then generates new, quote-like sentences.

### Project structure

- `data/quotes.txt` — training data (one quote per line).
- `src/config.py` — training and model hyperparameters.
- `src/tokenizer.py` — simple character-level tokenizer.
- `src/dataset.py` — dataset and dataloader utilities.
- `src/model.py` — a small Transformer (MiniGPT) language model.
- `src/train.py` — training loop that saves the model and tokenizer.
- `src/generate.py` — utilities to load the checkpoint and generate a quote.
- `src/zerochat.py` — a minimal ZeroChat-style conversation wrapper around the quote generator.
- `saved_models/model.pt` — model checkpoint (created after training).
- `main.py` — CLI entry point to train and generate quotes.
- `app.py` — Flask web server.
- `templates/index.html` — main HTML page for the web app.
- `static/style.css` — styles for the web UI.
- `static/app.js` — frontend logic for calling the API.

### Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows
source .venv/bin/activate  # on macOS/Linux
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Edit `data/quotes.txt` and add your own quotes (one per line).

### Training

Run:

```bash
python main.py train --max-iters 2000
```

This will:

- Read `data/quotes.txt`.
- Build a character-level tokenizer.
- Train a small GPT-style model.
- Save the best checkpoint to `saved_models/model.pt`.
- Save the tokenizer to `saved_models/tokenizer.json`.

You can adjust `--max-iters` for shorter or longer training runs.

### Generating quotes (CLI)

After training, generate a quote with:

```bash
python main.py generate
```

You can optionally provide a prompt:

```bash
python main.py generate --prompt "Life is"
```

This will print a single generated quote-like sentence to the console.

### Running the web app

1. Make sure you have trained a model at least once so that `saved_models/model.pt` and `saved_models/tokenizer.json` exist (see training section above).

2. Start the Flask server:

```bash
python app.py
```

3. Open your browser at `http://127.0.0.1:5000` to use the web interface:

- Enter an optional prompt (e.g. `Life is`).
- Click **Generate quote**.
- The generated quote will appear in the card below.

