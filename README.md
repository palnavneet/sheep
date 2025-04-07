<h1 align="center">🐑 Sheep: An Android Library for Running NLP Models</h1>

<h3 align="center" style="color:red;">🚧 Under Development</h3>

<p align="center">
  Sheep lets you run popular NLP models like DistilBERT directly on Android using ONNX Runtime.  
  It comes with built-in tokenizers (like WordPiece) and also supports custom pipelines and tokenizers out of the box.
</p>

---

## 🧠 Supported Models & Pipelines

| Model        | Type         | Pipelines Supported       | Tokenizer Support     | Status     |
|--------------|--------------|---------------------------|------------------------|------------|
| DistilBERT   | Transformer  | TextSimilarity, Custom    | WordPiece, Custom      | ✅ Working |
| (Coming Soon)| Classifier   | TextClassification        | WordPiece, SentencePiece| 🚧 Planned |
| (Coming Soon)| QA Model     | QuestionAnswering         | WordPiece              | 🚧 Planned |
| (Planned)    | GPT / Decoder| TextGeneration, Summarizer| BPE   

