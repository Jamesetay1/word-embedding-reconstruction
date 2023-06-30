# Hey! This Repo is a WIP but actively being updated over Summer 2023

# charCNN
Open source implementation of the Character-level CNN based on the architecture described in 
[Learning to Generate Word Representations using Subword Information](https://aclanthology.org/C18-1216/) (Kim et. al. 2019)


At a high level, this model is designed to learn subword embeddings that can be used to recreate word embeddings as they are needed.
This can be used both for creating embeddings for unknown (OOV) words, as well as so that models need not store millions (~2M in English) word embeddings, 
but can instead store <100 character level embeddings and the model to recreate the word embeddings as needed.

This repository was created as part of my Master's disseration, available [soon]

## Running Code



