# PyTorch_LM
PyTorch Implementation of Google Billion Word Language Model

# Results
* 52.68 Perplexity after 10 training epochs on a 4-layer, 512-unit LSTM Language Model
* Trained for 1 Week using 2 Nvidia Titan X Pascal GPUs
* Implemented [Sampled Softmax](https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss) and 
[Log-Uniform Sampler](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/log-uniform-candidate-sampler)

# Baseline Hyper-Parameters
| Parameter             | Value         |
| --------------------- | :----------:  |
| # Epochs              | 10            |
| Training Batch Size   | 128           |
| Evaluation Batch Size | 1             |
| BPTT                  | 20            |
| Embedding Size        | 512           |
| Hidden Size           | 512           |
| Tied                  | True          |
| # Layers              | 4             |
| Optimizer             | AdaGrad       |
| Learning Rate         | 0.01          |
| Gradient Clipping     | 1.0           |
| Dropout               | 0.10          |

# Updated Hyper-Parameters [1]
| Parameter                             | Value         |
| ------------------------------------- | :----------:  |
| Optimizer                             | AdaGrad       |
| Learning Rate                         | 0.1           |
| Gradient Clipping - LSTM Layers Only  | 10.0          |

I tied the word embedding and softmax weight matrices together to save GPU memory.
* [Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling](https://arxiv.org/pdf/1611.01462.pdf)

# Setup - Torch Data Format
1. Download Google Billion Word Dataset for Torch - [Link](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/leonardn/billionwords.tar.gz)
2. Run "process_gbw.py" on the "train_data.th7" file to create the "train_data.sid" file
3. Install Cython framework and build Log_Uniform Sampler

I leverage the GBW data preprocessed for the Torch framework. (See [Torch GBW](http://torch.ch/blog/2016/07/25/nce.html))
Each data tensor contains all the words in data partition. The "train_data.sid" file marks the start and end positions for each independent sentence.
The preprocessing step and "train_data.sid" file speeds up loading the massive training data. 

* Data Tensors - (test_data, valid_data, train_data, train_small, train_tiny) - (#words x 2) matrix - (sentence id, word id)
* Sentence ID Tensor - (#sentences x 3) matrix - (start position, end position, sentence length)

# Setup - Original Data Format
1. Download Google Billion Word Dataset - [Link](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz)

The Torch Data Format loads the entire dataset at once, so it requires at least 32 GB of memory.
The original format partitions the dataset into smaller chunks, but it runs slower.

# Resources
1. [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410) [Github](https://github.com/rafaljozefowicz/lm)
2. [Factorization Tricks for LSTM networks](https://arxiv.org/abs/1703.10722) [Github](https://github.com/okuchaiev/f-lm)
3. [Candidate Sampling](https://www.tensorflow.org/extras/candidate_sampling.pdf)
4. [Torch GBW](http://torch.ch/blog/2016/07/25/nce.html)
