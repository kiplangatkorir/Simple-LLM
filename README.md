# Simple LLM from Scratch

This repository demonstrates how to build and train a simple language model (LLM) from scratch using PyTorch. The model is trained on a small dataset and can generate text based on the learned patterns.


## Features
- **Simple LSTM-based Architecture**: The model uses an LSTM (Long Short-Term Memory) network for sequential data processing.
- **Text Tokenization**: Converts text into sequences of tokens for training.
- **Text Generation**: Generates text based on a given prompt.
- **Custom Dataset Support**: Easily replace the example text with your own dataset.

## Requirements
- Python 3.8+
- Libraries:
  - `torch`
  - `numpy`
  - `torchvision` (optional for visualization)
  - `tqdm` (for progress monitoring)

Install the required libraries:
```bash
pip install torch numpy tqdm
```

## How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/simple-llm.git
cd simple-llm
```

### 2. Prepare the Dataset
Replace the `text` variable in the script with your dataset. For example:
```python
text = "Once upon a time in a faraway land, there was a wise king..."
```

### 3. Run the Training Script
Execute the training script to train the model:
```bash
python train.py
```

### 4. Generate Text
After training, use the `generate_text` function to produce text. Provide a starting prompt and specify the length of the generated text. Example:
```python
print(generate_text(model, start_str="The king was ", length=100))
```

## Example Output
```plaintext
Epoch 1, Loss: 2.9607
Epoch 20, Loss: 0.1883
Generated Text:
The king was kind and just. Every day, he would sit upon his throne and listen to the needs and concerns of his people. One day, a young traveler arrived at the palace gates. He carried a strange artifact...
```

## Customization
1. **Model Architecture**:
   - Modify the `SimpleLSTM` class to use GRU or Transformer layers.
2. **Dataset**:
   - Replace the example text with a larger dataset for better results.
3. **Training Hyperparameters**:
   - Adjust `embed_dim`, `hidden_dim`, `seq_length`, or learning rate to optimize performance.

## Limitations
- **Small Datasets**: Works best with small datasets for experimentation.
- **Overfitting**: On tiny datasets, the model may memorize rather than generalize.
- **Limited Computational Power**: Designed for lightweight training.

## Future Work
- Implement advanced sampling techniques like top-k or nucleus sampling.
- Add support for multi-layer LSTM models.
- Transition to Transformer-based architectures for better scalability.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Inspired by basic neural network concepts and the desire to make LLMs accessible to everyone.

