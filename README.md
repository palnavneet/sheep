# Sheep

[![](https://jitpack.io/v/palnavneet/sheep.svg)](https://jitpack.io/#palnavneet/sheep) [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) ![PRs Welcome](https://img.shields.io/badge/PRs-welcome-green.svg)<br>
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

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).


