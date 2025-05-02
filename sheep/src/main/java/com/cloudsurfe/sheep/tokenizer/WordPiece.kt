package com.cloudsurfe.sheep.tokenizer

import android.content.Context
import android.util.Log
import com.cloudsurfe.sheep.core.Sheep.Companion.TAG
import java.io.BufferedReader
import java.io.InputStreamReader

class WordPiece(
    private val context: Context,
    private val assetVocabFileName: String
) : Tokenizer {

    private lateinit var idToToken: MutableMap<Int, String>
    private lateinit var tokenToId: MutableMap<String, Int>
    private val whitespaceRegex = "\\s+".toRegex()
    private var isVocabLoaded = false

    override fun loadVocab() {
        if (!isVocabLoaded) {
            idToToken = HashMap()
            tokenToId = HashMap()
            try {
                context.assets.open(assetVocabFileName).use { inputStream ->
                    BufferedReader(InputStreamReader(inputStream)).use { reader ->
                        reader.lineSequence().forEachIndexed { index, line ->
                            idToToken[index] = line
                            tokenToId[line] = index
                        }
                    }
                }
                isVocabLoaded = true
            } catch (e: Exception) {
                Log.d(TAG, "Failed to read $assetVocabFileName : ${e.message}")
            }
        }
    }

    override fun tokenize(inputText: String, isTokenTypeIds: Boolean): LongArray {
        if (!isVocabLoaded) loadVocab()
        val unkId = tokenToId.getOrDefault("[UNK]", 100)
        val clsId = tokenToId.getOrDefault("[CLS]", 101)
        val sepId = tokenToId.getOrDefault("[SEP]", 102)

        val words = inputText.trim().lowercase().split(whitespaceRegex)
        val inputIds = mutableListOf<Long>()
        inputIds.add(clsId.toLong())
        if (isTokenTypeIds) {
            val segments = inputText.trim().lowercase().split("\\s*\\[SEP]\\s*".toRegex(RegexOption.IGNORE_CASE), limit = 2)
            if (segments.size == 2) {
                segments[0].split(whitespaceRegex).forEach{ word ->
                    inputIds.add(tokenToId.getOrDefault(word, unkId).toLong())
                }
                inputIds.add(sepId.toLong())
                segments[1].split(whitespaceRegex).forEach{ word ->
                    inputIds.add(tokenToId.getOrDefault(word,unkId).toLong())
                }
                inputIds.add(sepId.toLong())

            } else {
                words.forEach{ word ->
                    inputIds.add(tokenToId.getOrDefault(word, unkId).toLong())
                }
                inputIds.add(sepId.toLong())
            }
        } else {
            words.forEach{ word ->
                inputIds.add(tokenToId.getOrDefault(word, unkId).toLong())
            }
            inputIds.add(sepId.toLong())
        }
        return inputIds.toLongArray()
    }

    fun getTokenTypeIds(input : LongArray) : IntArray{
        var sepCount = false
        return input.map { input ->
            val token = deTokenize(input.toInt())
            if (token == "[SEP]"){
                if (!sepCount){
                    sepCount = true
                    0
                }else{
                    1
                }
            }else{
                if (!sepCount) 0 else 1
            }
        }.toIntArray()
    }

    override fun deTokenize(tokenId: Int) = idToToken.getOrDefault(tokenId, "[UNK]")

}