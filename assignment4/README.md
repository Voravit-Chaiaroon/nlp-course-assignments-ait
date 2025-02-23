# Text Similarity Web App with Sentence-BERT

This repository demonstrates a simple Flask web application that leverages a custom-trained Sentence-BERT model—pretrained using BERT—to perform Natural Language Inference (NLI) for text similarity. The web app provides a user-friendly interface where users can input two sentences (a premise and a hypothesis) and receive a prediction indicating whether the relationship between them is Entailment, Neutral, or Contradiction.

![App Screenshot](pictures/app.png)

## Dataset For BERT Model: [BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus)

### Description

The BookCorpus dataset is a collection of over 11,000 books written by unpublished authors. It is widely used for training language models due to its diverse and extensive text data. The dataset is available in the Hugging Face datasets library, making it easy to load and use.

## Model Details

Base Architecture: BERT (Bidirectional Encoder Representations from Transformers)  
Pretraining: The Sentence-BERT model was pretrained on an NLI task using [SNLI datasets](https://huggingface.co/datasets/stanfordnlp/snli).  
The detailed model is:
- Number of Encoder of Encoder Layer      : 12
- Number of heads in Multi-Head Attention : 12
- Embedding Size                          : 768
- FeedForward dimension                   : 768 * 4
- Dimension of K, Q, and V                : 64
- Number of Segments                      : 2

## Evaluation

The S-BERT model was trained and evaluated on a natural language inference (NLI) task using the SNLI dataset. The evaluation results indicate that the model's performance is suboptimal, with a validation accuracy of 29%.

### Key Metrics

Validation Accuracy: 0.2900 (29%)

Classification Report:

- Precision: 0.10 (macro avg), 0.08 (weighted avg)
- Recall: 0.33 (macro avg), 0.29 (weighted avg)
- F1-Score: 0.15 (macro avg), 0.13 (weighted avg)

|               | precision | recall | f1-score | support |
| ------ | ------ | ------ | ------ | ------ |
| Entailment    | 0.00      | 0.00   | 0.00     | 33      |
| Contradiction | 0.00      | 0.00   | 0.00     | 38      |
| accuracy      | 0.29      | 1.00   | 0.45     | 29      |
| macro avg     | 0.10      | 0.33   | 0.15     | 100     |
| weighted avg  | 0.08      | 0.29   | 0.13     | 100     |

Confusion Matrix:  
[[0  0 33]  
[ 0  0 38]  
[ 0  0 29]]

## Limitations and Challenges

From the results we can tell that the model only predicts one class, it indicates a significant issue with the model's training or the model's architecture or the dataset.

- Class Imbalance or Data Issues:
    - The model may have been trained on a dataset with an imbalanced distribution of classes or with mislabeled examples. This could cause the model to overfit the majority class or ignore minority classes.

- Insufficient or Ineffective Fine-Tuning:
    - The training regime (e.g., number of epochs, learning rate, or optimizer settings) might not be optimal, leading the model to converge to a degenerate solution where it always predicts a single class.
    - The classifier head on top of the Sentence-BERT embeddings might be under-parameterized or not receiving a rich enough signal from the embedding layer.
- Pooling Strategy Limitations:
    - Mean pooling might not be the best way to capture the most discriminative features from the token embeddings, possibly leading to loss of important information required for differentiating between classes.

- Feature Extraction Issues:
    - The way the embeddings from the premise and hypothesis are combined (concatenation of mean-pooled vectors and their absolute difference) might not fully capture the relationship needed to discriminate between entailment, neutral, and contradiction.


## Potential Improvements & Modifications
- Data and Preprocessing Enhancements:
    - Balance the Dataset: Ensure that the training data is balanced across all classes. If necessary, apply techniques such as oversampling the minority classes or using class weighting during loss computation.
    - Data Augmentation: Incorporate additional training data or use data augmentation strategies to enrich the diversity of examples in each class.

- Model and Training Adjustments:
    - Hyperparameter Tuning: Experiment with different learning rates, batch sizes, and number of epochs. Use techniques like early stopping to prevent overfitting or underfitting.
    - Advanced Optimizers: Consider trying alternative optimizers (like AdamW) or adjusting the weight decay to improve convergence.
    - Loss Function Modification: Use a loss function that can handle class imbalance better (e.g., weighted cross-entropy loss) to encourage the model to learn underrepresented classes.

- Improving Feature Extraction:
    - Pooling Methods: Experiment with different pooling strategies such as max pooling or attention-based pooling instead of just mean pooling to better capture salient features from the embeddings.
    - Enhanced Classifier Head: Redesign the classifier head by adding additional fully connected layers, dropout layers, or even attention mechanisms to better capture the nuances between the premise and hypothesis.
    - Joint Encoding: Instead of encoding the premise and hypothesis separately and then combining their representations, consider using a joint encoding strategy where the input is the concatenation of both sentences (with appropriate segment embeddings) so that the model can directly capture their interaction.

- Error Analysis and Iterative Improvements:
    - Confusion Matrix Analysis: Perform a deeper error analysis to understand which types of examples are misclassified. This can help in understanding if certain classes are systematically confused with others.
    - Validation on Multiple Datasets: Validate the model on different subsets or related NLI datasets (like SNLI and MNLI) to assess generalizability and pinpoint domain-specific issues.

## Conclusion

The current confusion matrix reveals that the model is not differentiating between classes and always predicts the same label. Addressing this issue will likely involve improving the quality and balance of the training data, fine-tuning the training process, and refining the model architecture (especially the classifier head and pooling strategy). By iteratively experimenting with these adjustments and closely monitoring evaluation metrics, you can work towards a more balanced and accurate model for the NLI task.


### Citation

Book Corpus Dataset

```python
@article{zhu2015aligning,
  title={Aligning books and movies: Towards story-like visual explanations by watching movies and reading books},
  author={Zhu, Yukun and Kiros, Ryan and Zemel, Rich and Salakhutdinov, Ruslan and Torralba, Antonio and Urtasun, Raquel and Fidler, Sanja},
  journal={arXiv preprint arXiv:1506.06724},
  year={2015}
}
```

SNLI Dataset

```python
@InProceedings{Zhu_2015_ICCV,
    title = {Aligning Books and Movies: Towards Story-Like Visual Explanations by Watching Movies and Reading Books},
    author = {Zhu, Yukun and Kiros, Ryan and Zemel, Rich and Salakhutdinov, Ruslan and Urtasun, Raquel and Torralba, Antonio and Fidler, Sanja},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {December},
    year = {2015}
}
```
