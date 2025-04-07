package com.cloudsurfe.sheep.tokenizer

sealed class TokenizerType {
    data object  WordPiece : TokenizerType()
    data class CustomTokenizer(val tokenizer: Tokenizer) : TokenizerType()
}