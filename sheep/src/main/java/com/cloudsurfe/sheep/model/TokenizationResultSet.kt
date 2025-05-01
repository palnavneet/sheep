package com.cloudsurfe.sheep.model

data class TokenizationResultSet(
    val input : List<String>,
    val tokenizedInput : List<LongArray>,
    val detokenizedInput : List<String>
)