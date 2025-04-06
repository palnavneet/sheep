package com.cloudsurfe.sheep.core

interface NLP {
    fun run(assetFileName: String, pipeline: Pipeline, tokenizerType: TokenizerType)
}