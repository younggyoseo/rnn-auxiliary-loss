# rnn-auxiliary-loss

An Implementation of [Learning Longer-term Dependencies in RNNs with Auxiliary Losses](https://arxiv.org/abs/1803.00144) (Trinh et al. 2018) in PyTorch.

The paper proposes a simple method to augment RNNs with unsupervised auxiliary losses in order to improve their ability to capture long-term dependencies.

<img src="https://raw.githubusercontent.com/belepi93/rnn-auxiliary-loss/master/pics/overview.png" width="500">
<img src="https://raw.githubusercontent.com/belepi93/rnn-auxiliary-loss/master/pics/r-LSTM.png" width="500">
<img src="https://raw.githubusercontent.com/belepi93/rnn-auxiliary-loss/master/pics/result.png" width="500">

This repo gives you an **incomplete** implementation of LSTM augmented with reconstruction auxiliary loss(r-LSTM). Since i was not able to find any code available for this paper, i had to improvise many details by myself. I tried to reproduce paper's results but without success. I'm really waiting for any comments or contributions to improve this repo. Thanks!

# Requirements

```
PyTorch 0.4 & Python 3.6
Numpy
Torchvision
TensorboardX
```

# Examples

`python main.py --cuda` for full training with BPTT 300.

`python main.py --cuda --bptt 784` for full training with full BPTT.

`python main.py --cuda --single` for LSTM.

`python main.py --cuda --pre_epochs 0` for skipping pretraining.

`python main.py --cuda --dataset MNIST` to use MNIST as dataset.

# Dataset

You can use MNIST or pMNIST with `--dataset MNIST` or `--dataset pMNIST`.

pMNIST is sequential MNIST where each pixel sequence is permuted in the same. It is harder to capture long-term dependencies in pMNIST so the efficacy of using r-LSTM stands out much more when using pMNIST

# pMNIST Benchmark Results

| Models            | No Emb | Full Emb | Part Emb |
|:-----------------:|:----------:|:----------:|:----------:|
| LSTM Full BP  |0.9095 | 0.8406 | 0.8759 |
| LSTM Truncate 300 |0.9026 | 0.841 | 0.873 |
| r-LSTM T300   |0.9037 | 0.8743 | **0.8863** |
| r-LSTM Full BP    |**0.9129** | **0.8856** | 0.8835 |

I'm not sure how to implement embedding(project input) in a paper so tried to test several methods. In `No Emb`, I just tried to 1 dimensional input pixel to 128 dimensional dense vector without embedding matrix. In `Full Emb`, Every pixel(0~255) has its own dense vector in embedding matrix. In `Part Emb`, Embedding is only applied to auxiliary decoder network.

