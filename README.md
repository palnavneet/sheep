<h1 align="center">🐑 Sheep: An Android Library for Running NLP Models</h1>

<h3 align="center" style="color:red;">🚧 Under Development</h3>

<p align="center">
  <img src="https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif" width="250"/>
</p>

<p align="center">
  Sheep lets you run popular NLP models like DistilBERT directly on Android using ONNX Runtime.  
  It comes with built-in tokenizers (like WordPiece) and also supports custom pipelines and tokenizers out of the box.
</p>

---

| 🧠 Model       | ⚙️ Type         | 🧩 Pipelines Supported         | 🔤 Tokenizer Support         | 📊 Status       |
|---------------|------------------|-------------------------------|------------------------------|-----------------|
| [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) | Transformer       | TextSimilarity, Custom        | WordPiece, Custom            | ✅ Working       |
| [BERT (Planned)](https://huggingface.co/bert-base-uncased) | Classifier        | TextClassification            | WordPiece, SentencePiece     | 🚧 Planned       |
| [RoBERTa (Planned)](https://huggingface.co/roberta-base) | QA Model          | QuestionAnswering             | WordPiece                    | 🚧 Planned       |
| [GPT-2 (Planned)](https://huggingface.co/gpt2) | Decoder           | TextGeneration, Summarizer    | BPE                          | 🧪 In Design     |

