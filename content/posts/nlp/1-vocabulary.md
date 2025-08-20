---
author: "Le Van Dong"
title: "Vocabulary in NLP"
date: "2025-08-20"
tags: ["AI", "Machine Learning"]
categories: ["Natural Language Processing"]
---

## **Overview of Natural Language Processing (NLP)**

Human language exists primarily in two forms: speech and text. Speech is the basis for applications like Siri and Google Assistant, while text data is the core input for most NLP tasks. A fundamental challenge in NLP is that computers cannot directly understand human language. Therefore, text must be converted from a string of characters into a numerical format that a computer can process. This conversion is a crucial preprocessing step for machine learning models.

## **1. Corpus and Vocabulary**

- **Text Corpus:** A large collection of texts used for a specific purpose. For example, if you want to build a news classification model, the collection of all articles used to train and evaluate the model is the text corpus.
- **Vocabulary:** A set of all the *unique* words that appear in the text corpus. Building and using this vocabulary is a core step in most NLP pipelines.

For example, with the corpus "Tôi yêu AI. AI rất tuyệt." (I love AI. AI is great.), the vocabulary would be: `{"Tôi", "yêu", "AI", "rất", "tuyệt"}`.

## **2. Tokenization**

Before building the vocabulary, raw text needs to be split into smaller, more basic units.

- **Token:** The smallest unit of text that the model will process. Depending on the chosen method, a token can be a word, a character, or a subword (prefix/suffix).
- **Tokenization:** The process of splitting a text string into a list of tokens. This is the first and most important step in data preparation.

## **3. Building the Vocabulary (Dictionary)**

After creating a vocabulary from the corpus, the next step is to build a "dictionary" that maps each token to a unique numerical value (typically an integer ID). This process is similar to how humans learn and memorize new words. Machine learning models can only "understand" the tokens that are in their dictionary. Words that have never appeared before are considered "out-of-vocabulary" (OOV) or unknown words.

The process of creating a dictionary generally follows these rules:

1. **Frequency Statistics:** Count the occurrences of each token in the entire corpus.
2. **Sorting:** Sort the tokens in descending order of their frequency. If two tokens have the same frequency, they are sorted alphabetically.
3. **ID Assignment:** Assign a sequential integer ID to each token in the sorted order. Typically, ID `0` is reserved for padding or for OOV tokens.

### **Illustrative Example**

Given a text corpus with two sentences: `["bob ate apples and pears", "fred ate apples!"]`

**Step 1: Preprocessing and Tokenization**

- Convert to lowercase and remove punctuation.
- The processed corpus is: `["bob", "ate", "apples", "and", "pears", "fred", "ate", "apples"]`

**Step 2: Building the Vocabulary and Counting Frequencies**

- Vocabulary: `{"bob", "ate", "apples", "and", "pears", "fred"}`
- Frequencies:
  - ate: 2
  - apples: 2
  - bob: 1
  - pears: 1
  - fred: 1
  - and: 1

**Step 3: Sorting and Creating the Dictionary**

- Sort by descending frequency. Since "ate" and "apples" have the same frequency, they are sorted alphabetically. The same applies to the remaining words.
- Priority order: `(ate: 2), (apples: 2), (bob: 1), (pears: 1), (fred: 1), (and: 1)`
- Assign IDs (starting from 1, reserving 0 for OOV tokens):

```json
{
  "ate": 1,
  "apples": 2,
  "bob": 3,
  "pears": 4,
  "fred": 5,
  "and": 6
}
```

**Step 4: Encoding New Text**
Now, a new sentence can be converted into a sequence of numbers using this dictionary:

- New text: `["bob ate pears", "fred ate pears too"]`
- The word "cũng" is not in the dictionary, so it is treated as an OOV token and receives ID `0`.
- The numerical representation is: `[[3, 1, 4], [5, 0, 1, 4]]`

This successfully converts the text data into a numerical format, making it ready to be used as input for NLP models.
