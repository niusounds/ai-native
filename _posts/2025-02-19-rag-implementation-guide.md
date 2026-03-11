---
layout: post
title: "RAG（Retrieval-Augmented Generation）実装ガイド：社内ドキュメントをLLMで検索可能にする"
description: "RAGシステムの仕組みから実装まで、社内ドキュメントをLLMで効率的に検索・回答するシステムを構築する方法を詳しく解説します。"
date: 2025-02-12 10:00:00 +0900
categories: [llm]
tags: [RAG, ベクトルDB, LlamaIndex, embeddings, 実装]
author: "AI Native Engineer"
reading_time: 14
---

## RAGとは何か

RAG（Retrieval-Augmented Generation）は、LLMの知識の限界を克服するためのアーキテクチャパターンです。

```
従来のLLM:
ユーザー質問 → LLM（学習データのみ） → 回答

RAG:
ユーザー質問 → ベクトル検索（関連ドキュメント取得） → LLM（取得情報+質問） → 根拠ある回答
```

**RAGが解決する問題:**
- 学習データカットオフ以降の情報への対応
- 社内固有情報・機密情報の活用
- ハルシネーション（事実誤認）の削減
- 回答の根拠となる出典の提示

## システムアーキテクチャ

```
[ドキュメント処理フェーズ]
PDFs/Docs/URLs
      ↓
  テキスト抽出
      ↓
  チャンキング（分割）
      ↓
  埋め込み生成（Embedding）
      ↓
  ベクトルDB（保存）

[クエリフェーズ]
ユーザー質問
      ↓
  クエリの埋め込み生成
      ↓
  類似度検索（ベクトルDB）
      ↓
  関連チャンクの取得
      ↓
  LLMへのプロンプト生成
      ↓
  最終回答
```

## セットアップ

```bash
pip install llama-index llama-index-embeddings-openai \
    llama-index-llms-openai chromadb pypdf
```

## 基本的なRAGシステム

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# LLMと埋め込みモデルの設定
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# ドキュメントの読み込み
documents = SimpleDirectoryReader("./docs").load_data()

# インデックスの作成（ベクトルDBへの保存）
index = VectorStoreIndex.from_documents(documents)

# クエリエンジンの作成
query_engine = index.as_query_engine(
    similarity_top_k=3,  # 上位3件の類似ドキュメントを取得
    response_mode="compact",
)

# 質問
response = query_engine.query("有給休暇の申請手順を教えてください")
print(response)

# ソース（根拠）の確認
for node in response.source_nodes:
    print(f"ソース: {node.metadata.get('file_name', 'unknown')}")
    print(f"スコア: {node.score:.3f}")
    print(f"テキスト: {node.text[:200]}...")
    print("---")
```

## 永続化：ChromaDBの使用

毎回インデックスを再作成するのは非効率です。ChromaDBを使って永続化します。

```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, load_index_from_storage
import os

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "company_docs"

def get_or_create_index(docs_path: str):
    """既存のインデックスを読み込むか、新規作成する"""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 既存データがあれば読み込み
    if chroma_collection.count() > 0:
        print(f"既存インデックスを読み込み中... ({chroma_collection.count()}件のドキュメント)")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(vector_store)
    
    # 新規作成
    print("新しいインデックスを作成中...")
    documents = SimpleDirectoryReader(docs_path).load_data()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print(f"インデックス作成完了: {len(documents)}件のドキュメント")
    return index

index = get_or_create_index("./company_docs")
```

## チャンキング戦略の最適化

ドキュメントの分割方法（チャンキング）はRAGの精度に大きく影響します。

```python
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding

# 方法1: 固定サイズチャンキング（シンプル）
fixed_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
)

# 方法2: セマンティックチャンキング（推奨）
# 意味的に類似したテキストをまとめてチャンクを作成
semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=OpenAIEmbedding(),
)

# セマンティックチャンキングの使用
documents = SimpleDirectoryReader("./docs").load_data()
nodes = semantic_splitter.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)
```

## ハイブリッド検索の実装

ベクトル検索（意味検索）とキーワード検索を組み合わせることで精度が向上します。

```python
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# ベクトル検索リトリーバー
vector_retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)

# BM25（キーワード）検索リトリーバー
bm25_retriever = BM25Retriever.from_defaults(
    index=index,
    similarity_top_k=5,
)

# ハイブリッドリトリーバー（両者を統合）
hybrid_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=3,
    num_queries=1,  # クエリ拡張なし（1のみ）
    mode="reciprocal_rerank",  # Reciprocal Rank Fusion
    use_async=True,
)

# クエリエンジンの作成
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    llm=Settings.llm,
)
```

## RAGの評価方法

RAGシステムの品質を定量的に測定するのは重要です。

```python
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
)

# 評価器の設定
faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)

# テストケース
test_questions = [
    "有給休暇は年間何日もらえますか？",
    "在宅勤務の申請方法を教えてください",
    "経費精算の締め切りはいつですか？",
]

async def evaluate_rag(questions: list[str]):
    results = []
    for question in questions:
        response = await query_engine.aquery(question)
        
        # 忠実性の評価（回答がソースに基づいているか）
        faithfulness_result = await faithfulness_evaluator.aevaluate_response(
            response=response
        )
        
        # 関連性の評価（回答が質問に関連しているか）
        relevancy_result = await relevancy_evaluator.aevaluate_response(
            query=question,
            response=response,
        )
        
        results.append({
            "question": question,
            "answer": str(response),
            "faithfulness": faithfulness_result.passing,
            "relevancy": relevancy_result.passing,
        })
    
    return results

import asyncio
results = asyncio.run(evaluate_rag(test_questions))

# 評価レポート
for r in results:
    print(f"質問: {r['question'][:50]}...")
    print(f"忠実性: {'✅' if r['faithfulness'] else '❌'}")
    print(f"関連性: {'✅' if r['relevancy'] else '❌'}")
    print()
```

## 本番環境での考慮事項

### コスト最適化

```python
# 小さいモデルをリトリーバーに、大きいモデルを回答生成に使い分ける
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",  # 安い埋め込みモデル
)
Settings.llm = OpenAI(
    model="gpt-4o-mini",  # 軽いモデル（精度に問題なければ）
)
```

### キャッシング

```python
from llama_index.core.storage.kvstore import SimpleKVStore
from llama_index.core import set_global_handler

# レスポンスキャッシュ（同じ質問への再計算を防ぐ）
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_query(question_hash: str, question: str) -> str:
    response = query_engine.query(question)
    return str(response)

def smart_query(question: str) -> str:
    question_hash = hashlib.md5(question.encode()).hexdigest()
    return cached_query(question_hash, question)
```

## まとめ

RAGシステムの構築に必要な要素をカバーしました：

1. **基本的なRAGパイプライン**の実装
2. **永続化**（ChromaDB）
3. **チャンキング戦略**の最適化
4. **ハイブリッド検索**による精度向上
5. **評価フレームワーク**の構築
6. 本番環境での**コスト最適化**

RAGは「作って終わり」ではなく、評価→改善のサイクルが重要です。まず動くものを作り、評価指標を設定して継続的に改善していきましょう。
