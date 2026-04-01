# Reimplement Transformer Model from Scratch

This project aims to reimplement the Transformer model architecture from scratch using PyTorch, with modular code for each component. 

## Project Structure

- `src/`
  - `attention.py`: **(To be implemented)** Multi-head self-attention and related attention mechanisms.
  - `embeddings.py`: Contains `PositionalEncoding` and `TransformerEmbedding` modules (done).
  - `layers.py`: **(To be implemented)** Transformer encoder/decoder layers, feed-forward, layer norm, etc.
  - `model.py`: **(To be implemented)** The main Transformer model class, assembling all components.
  - `tokenizer.py`: Implements a wrapper for BERT's tokenizer (done).
  - `utils.py`: **(To be implemented)** Utility functions (masking, etc.).
- `train.py`: **(To be implemented)** Training loop, loss, optimizer, evaluation, etc.
- `tests/`: Unit tests for each module (to be filled in).
- `data/`: Place your datasets here.
- `requirements.txt`: Python dependencies.

## What’s Done
- `embeddings.py` and `tokenizer.py` are implemented and can be used as references for style and structure.

## What Needs to be Implemented
- `attention.py`: Implement multi-head self-attention and scaled dot-product attention.[Bui Nhat Tan]
- `layers.py`: Implement encoder and decoder layers, including feed-forward and normalization.[Nguyen Minh Quan]
- `model.py`: Assemble the full Transformer model using the above modules.
- `utils.py`: Add helper functions (e.g., for creating masks).
- `train.py`: Write the training loop, loss calculation, optimizer setup, and evaluation logic.
- `tests/`: Add unit tests for all modules.

## How to Contribute
1. **Pick a module** from the "What Needs to be Implemented" list.
2. **Follow the style** in `embeddings.py` and `tokenizer.py` (docstrings, comments, modularity).
3. **Write clear docstrings** and comments for all classes and functions.
4. **Test your code**: Add or update tests in `tests/`.
5. **Document any design decisions** in this README if needed.

## Setup
1. Clone the project
    ```bash
   git clone https://github.com/Vupiad/Reimplement-Transformer-model.git

   cd Reimplement-Transformer-model
   
   ```
2. Create a virtual environment
    ```bash
   python -m venv myenv

   ./myenv/Scripts/Activate.ps1

   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Let's get started!


## References
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

---
**Please update this README as you implement new parts!**
