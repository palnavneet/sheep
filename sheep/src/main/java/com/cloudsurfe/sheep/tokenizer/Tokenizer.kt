package com.cloudsurfe.sheep.tokenizer

interface Tokenizer {
    fun loadVocab()
    fun tokenize(inputText: String): LongArray
    fun deTokenize(tokenId: Int): String
}