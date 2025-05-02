package com.cloudsurfe.sheep.tokenizer

interface Tokenizer {
    fun loadVocab()
    fun tokenize(inputText: String, isTokenTypeIds : Boolean = false): LongArray
    fun deTokenize(tokenId: Int): String
}