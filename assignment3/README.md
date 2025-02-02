# Machine Translation (MT) with Transformers
Machine Translation (MT) is the task of automatically converting text from one language to another. Transformers, introduced in the "Attention is All You Need" paper, have revolutionized MT by using self-attention mechanisms to capture long-range dependencies in text.  Compared to traditional RNN-based approaches, Transformers process entire sequences in parallel, leading to faster and more accurate translations.

## 1. Get Language Pair

- **Datasets Pair of Thai and English**
  Datasets from [scb-mt-en-th-2020](https://github.com/vistec-AI/dataset-releases/releases/tag/scb-mt-en-th-2020_v1.0) is used to train this model. 
  ### Description: 
  AI Research Institute of Thailand (AIResearch), with the collaboration between Vidyasirimedhi Institute of Science and Technology (VISTEC) and Digital Economy Promotion Agency (depa), publishes an open English-Thai machine translation dataset, with the sponsorship from Siam Commercial Bank (SCB), namely scb-mt-en-th-2020. The dataset contains parallel sentences from various sources such as task-based conversation, organization websites, Wikipedia articles, and government documents.

  To obtain parallel sentences, we hire professional and crowdsourced translators and build a module to automatically align parallel sentence pairs from documents, articles, and web pages.

- **Preparing the Dataset for the Translation Model**

  This section outlines the detailed process of preparing the dataset for use in a translation model, with a focus on Thai language-specific requirements. The steps include text normalization, tokenization, and word segmentation, leveraging tools like _PyThaiNLP_ for effective processing.

  Main Tools:
  [PyThaiNLP](https://pythainlp.org/dev-docs/api/tokenize.html): A powerful library for Thai NLP tasks, which includes functions for Thai-specific normalization, such as removing zero-width spaces and normalizing Thai numerals.

  ### Normalization
  Text normalization ensures that the text is in a consistent and clean format, making it easier to process. For Thai text, this involves:
    - Removing unnecessary whitespace: Extra spaces between words or sentences are eliminated.
    - Standardizing characters: Ensuring consistent use of characters (e.g., full-width vs. half-width characters).
    - Handling special characters: Removing or replacing irrelevant special characters.

  ### Tokenization
  Tokenization splits text into smaller units, such as words or subwords. For Thai, tokenization is particularly challenging because the language does not use spaces to separate words. Which _PyThaiNLP_: This library provides multiple tokenization engines, such as newmm (default), longest, and deepcut, to handle Thai word segmentation effectively.

## 2. Evaluation  and Verification 

- **Performance comparison of these attention mechanisms**
 From the results of model training, it might be hard to compare accuracy causing by fault of the equation of attention. But on other aspects we can compare the characteristics of the attentions in this assigment:
  - *General Attention* : Most simple and computationally efficient of all three.
  - *Multiplicative Attention* : More flexible than general attention due to the learnable weight matrix W and more capable to capture more complex relationships between source and target words.
  - *Additive Attention* : More expressive than multiplicative attention due to the non-linearity (tanh) makes it the most computationally expensive.

- **Training and validation loss plots**:
  ![App Screenshot](plotloss.png)

- **Training and validation loss**:
  ![App Screenshot](table.png)

- **Attention maps**:
  ![App Screenshot](attentionMap.png)

- **Analysis** 
Thai is a tonal language with complex grammatical structures. Additive Attention, with its non-linearity (tanh function), can potentially better capture the intricate relationships between words and their contexts within a sentence. Also Thai sentences can exhibit long-distance dependencies, where the meaning of a word is influenced by words far away in the sentence. Additive Attention, with its capacity to model complex interactions, might be better equipped to handle these dependencies.


## 3. Sample Capture of Application
  ![App Screenshot](app.png)