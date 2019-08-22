# PyTorch Large-Scale Language Model
A Large-Scale PyTorch Language Model trained on the 1-Billion Word (LM1B) / (GBW) dataset

# Latest Results
* **39.98 Perplexity** after 5 training epochs using LSTM Language Model with Adam Optimizer
* Trained in ~26 hours using 1 Nvidia V100 GPU (**~5.1 hours per epoch**) with 2048 batch size (**~10.7 GB GPU memory**)

# Previous Results
* **46.47 Perplexity** after 5 training epochs on a 1-layer, 2048-unit, 256-projection LSTM Language Model [3]
* Trained for 3 days using 1 Nvidia P100 GPU (**~12.5 hours per epoch**)
* Implemented [Sampled Softmax](https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss) and 
[Log-Uniform Sampler](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/log-uniform-candidate-sampler) functions

# GPU Hardware Requirement
| Type                 | LM Memory Size | GPU                            |
| -------------------- | -------------- | ------------------------------ |
| w/o tied weights     | ~9 GB          | Nvidia 1080 TI, Nvidia Titan X |
| w/ tied weights [6]  | ~7 GB          | Nvidia 1070 or higher          |

* There is an option to tie the word embedding and softmax weight matrices together to save GPU memory.

# Hyper-Parameters [3]
| Parameter                     | Value         |
| ----------------------------- | ------------- |
| # Epochs                      | 5             |
| Training Batch Size           | 128           |
| Evaluation Batch Size         | 1             |
| BPTT                          | 20            |
| Embedding Size                | 256           |
| Hidden Size                   | 2048          |
| Projection Size               | 256           |
| Tied Embedding + Softmax      | False         |
| # Layers                      | 1             |
| Optimizer                     | AdaGrad       |
| Learning Rate                 | 0.10          |
| Gradient Clipping             | 1.00          |
| Dropout                       | 0.01          |
| Weight-Decay (L2 Penalty)     | 1e-6          |

# Setup - Torch Data Format
1. Download Google Billion Word Dataset for Torch - [Link](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/leonardn/billionwords.tar.gz)
2. Run "process_gbw.py" on the "train_data.th7" file to create the "train_data.sid" file
3. Install Cython framework and build Log_Uniform Sampler
4. Convert Torch data tensors to PyTorch tensor format (Requires Pytorch v0.4.1)

I leverage the GBW data preprocessed for the Torch framework. (See [Torch GBW](http://torch.ch/blog/2016/07/25/nce.html))
Each data tensor contains all the words in data partition. The "train_data.sid" file marks the start and end positions for each independent sentence.
The preprocessing step and "train_data.sid" file speeds up loading the massive training data. 

* Data Tensors - (test_data, valid_data, train_data, train_small, train_tiny) - (#words x 2) matrix - (sentence id, word id)
* Sentence ID Tensor - (#sentences x 2) matrix - (start position, sentence length)

# Setup - Original Data Format
1. Download 1-Billion Word Dataset - [Link](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz)

The Torch Data Format loads the entire dataset at once, so it requires at least 32 GB of memory.
The original format partitions the dataset into smaller chunks, but it runs slower.

# References
1. [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410) [Github](https://github.com/rafaljozefowicz/lm)
2. [Factorization Tricks for LSTM networks](https://arxiv.org/abs/1703.10722) [Github](https://github.com/okuchaiev/f-lm)
3. [Efficient softmax approximation for GPUs](https://arxiv.org/abs/1609.04309) [Github](https://github.com/facebookresearch/adaptive-softmax)
4. [Candidate Sampling](https://www.tensorflow.org/extras/candidate_sampling.pdf)
5. [Torch GBW](http://torch.ch/blog/2016/07/25/nce.html)
6. [Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling](https://arxiv.org/pdf/1611.01462.pdf)
