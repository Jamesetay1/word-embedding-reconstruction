# Word Embedding Reconstruction with Subword Embeddings for Language Modelling

This repository is the main hub of work my MSc Dissertation "Word Embedding Reconstruction with Subword Embeddings for Language Modelling", available 

## Repository Contents
This repository is a mix of my own work and forks from other repositories.  

`charCNN` is a reimplementation of [Learning to Generate Word Representations using Subword Information](https://github.com/kamigaito/rnnlm-pytorch), which I use as a reconstruction method.

`cr_clean` is a fork of [Compact Reconstruction](https://github.com/losyer/compact_reconstruction), which I use as a reconstruction method.

`pyTorch_LM` is a fork of a [word level LM pyTorch tutorial](https://github.com/pytorch/examples/blob/main/word_language_model/README.md),
which I include the ability to use pre-trained embeddings and then use to build a language model and evaluate perplexity of various embeddings.

`distance_measuring` has inital tests of the quality of the reconstructed embeddings compared to reference embeddings, such as cosine similarity and P@n

`results/embedding_comparison` is where I put my results!

`utils` has a few repo-wide scripts for making baseline embeddings, limiting the character set in embeddings, etc.
