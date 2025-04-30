<h1 align="center">🐑 Sheep: An Android Library for Running NLP Models</h1>

<h3 align="center" style="color:red;">🚧 Under Development</h3>

<p align="center">
  <img src="https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif" width="250"/>
</p>


<p align="center">
Sheep-NLP-Android is an open-source Kotlin-based library designed to integrate advanced Natural Language Processing (NLP)
capabilities into Android applications using ONNX Runtime. It provides developers with easy access to pre-trained NLP models, 
allowing for tasks like tokenization, sentiment analysis, text classification, and more on Android devices.
</p>

<br>

## Key Features

- **ONNX Runtime Integration**: Efficiently run quantized NLP models on Android devices.
- **Custom Tokenizers**: Includes built-in tokenizers like WordPiece, with support for custom tokenizers.
- **Lightweight and Fast**: Optimized for mobile devices with minimal memory and CPU usage.
- **Pre-trained Model Support**: Run popular NLP models, including DistilBERT, for a variety of tasks.
- **Custom Pipelines**: Supports easy integration of custom NLP pipelines for specific use cases.
- **Open-Source**: Fully open-source and easy to extend for your specific needs.

---


| 🧠 Model       | ⚙️ Type         | 🧩 Pipelines Supported         | 🔤 Tokenizer Support         | 📊 Status       |
|---------------|------------------|-------------------------------|------------------------------|-----------------|
| [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) | Transformer       | TextSimilarity, Custom        | WordPiece, Custom            | ✅ Working       |
| [BERT (Planned)](https://huggingface.co/bert-base-uncased) | Classifier        | TextClassification            | WordPiece, SentencePiece     | 🚧 Planned       |
| [RoBERTa (Planned)](https://huggingface.co/roberta-base) | QA Model          | QuestionAnswering             | WordPiece                    | 🚧 Planned       |
| [GPT-2 (Planned)](https://huggingface.co/gpt2) | Decoder           | TextGeneration, Summarizer    | BPE                          | 🧪 In Design     |

---

<br>

## ⚙️ Basic Usage

```kotlin
val sheep = Sheep(
    context = context,
    pipeline = PipelineType.TextSimilarity("Hello", "World"),
    tokenizer = TokenizerType.WordPiece
)

sheep.run(
    assetModelFileName = "distilbert.onnx",
    assetVocabFileName = "vocab.txt"
)
```

<br>

### 🚧 3. **Roadmap**

```md
## 🚧 Roadmap

- [x] DistilBERT support
- [x] Text similarity pipeline
- [x] WordPiece tokenizer
- [ ] Add text classification pipeline
- [ ] Add GPT-style decoder model support
- [ ] SentencePiece / BPE tokenizer
- [ ] Upload to Maven Central
```

<br>

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


