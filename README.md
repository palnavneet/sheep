# Sheep

![Under Development](https://img.shields.io/badge/Status-Under%20Development-orange.svg)
[![](https://jitpack.io/v/palnavneet/sheep.svg)](https://jitpack.io/#palnavneet/sheep) [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) <br>
![Maintenance](https://img.shields.io/badge/Maintained%3F-Yes-brightgreen.svg) 

Sheep is a Kotlin-based Android library for clean and efficient natural language processing (NLP) using ONNX format models. Powered by [ONNX Runtime](https://onnxruntime.ai/), it supports a variety of pipelines including classification, question answering, and translation—all running fully offline. Sheep handles tokenization automatically, so you can focus on high-level tasks without worrying about low-level text preprocessing. The library is lightweight, fast, and designed with privacy and mobile performance in mind.

> Models must be in **ONNX format**. Quantized models are recommended for best performance on mobile devices.

## Supported Models

- [x] BERT
- [x] DistilBERT
- [ ] RoBERTa
- [ ] ALBERT
- [ ] T5
- [ ] MiniLM

## Supported Pipelines

- [x] Text Classification
- [x] Question Answering
- [ ] Token Classification (NER, POS)
- [ ] Sentiment Analysis
- [ ] Zero-shot Classification
- [ ] Fill-Mask
- [ ] Summarization
- [ ] Translation
- [ ] Text Generation
- [ ] Conversational
- [ ] Table Question Answering
- [ ] Multiple Choice
- [ ] Text-to-Speech (TTS)

## Quick Start

### Requirements

- Android minSdkVersion: `24` 
- Java: Version 11 or above
- Gradle: Version 7.0 or above
- **JitPack repository**: Add the following to your root `settings.gradle.kts` file:
  
```gradle
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}
```
- **Dependecy**: Add the following to your app-level `build.gradle.kts` file:
```gradle
dependencies {
    implementation("com.github.palnavneet:sheep:<version>") // Recommended: Replace <version> with the latest version
}
```

## Operations

you're now ready to start working with Sheep.

### Loading a Model

To load a model, use the Sheep class, defining variables for the model name and vocab file. Ensure the model is in the ONNX format and placed in the assets directory. For example, to load a DistilBERT model with WordPiece tokenization, use the following code:

```kotlin
// Define model name and vocab file
val modelName = "distilbert_classification_quantized.onnx"
val vocabFile = "vocab.txt"

// Create an instance of Sheep
val sheep = Sheep(
    this,
    TokenizerType.WordPiece,
    modelName,
    vocabFile
)
```

### Running a Text Classification Pipeline

Once your model is loaded, you can easily run a text classification task. Below is an example of how to classify a text input:

#### Single Input

```kotlin
val inputText = "I had a wonderful day at the park!"
val label = sheep.run("input" to inputText)
val predictedLabel = label[0]["predicted_label"]
val confidence = label[0]["confidence"]
Log.d("Sheep", "$predictedLabel")
Log.d("Sheep", "$confidence")
```

#### Multiple Inputs

```kotlin
val inputList = listOf("I had a wonderful day at the park!", "I'm feeling really down today", "This is the best movie I've seen!")
val label = sheep.run("input" to inputList)

val predictedLabel = label[0]["predicted_label"]
val confidence = label[0]["confidence"]
Log.d("sheep", "$predictedLabel")
Log.d("sheep", "$confidence")

val predictedLabel1 = label[1]["predicted_label"]
val confidence1 = label[1]["confidence"]
Log.d("sheep", "$predictedLabel1")
Log.d("sheep", "$confidence1")

val predictedLabel2 = label[2]["predicted_label"]
val confidence2 = label[2]["confidence"]
Log.d("sheep", "$predictedLabel2")
Log.d("sheep", "$confidence2")
```
### Running a Question Answering Pipeline

You can use Sheep to answer questions based on a provided context. Here’s an example of how to use the question-answering pipeline:

#### Single Input

```kotlin
val label = sheep.run("input" to Pair("What is the color of the sky?", "The color of the sky is blue"))
val answer = label[0]["answer"]
Log.d("sheep", "$answer")
```


## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).


