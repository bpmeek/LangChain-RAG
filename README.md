# LangChain-RAG

## Overview
This repository is meant to show an implementation of using Retrieval Augmented Generation (RAG) to increase the 
accuracy and detail of a small LLM

This repo uses `google/flan-t5-base` as its base model, this can be changed in `configs/base/parameters.yaml`

## Installing requirements

run the following to install requirements

```
pip install -r requirements.txt
```

## Examples

```
Question: What songs is Taylor Swift known for?
Base answer: "Sweetest"
Answer with RAG: "Teardrops on My Guitar"
```