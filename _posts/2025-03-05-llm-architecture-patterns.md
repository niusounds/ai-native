---
layout: post
title: "AI時代のソフトウェアアーキテクチャ：LLMを組み込んだシステム設計のパターンと落とし穴"
description: "LLMをプロダクションシステムに組み込む際の設計パターン、信頼性確保の方法、コスト管理、モニタリングについて実践的な視点で解説します。"
date: 2025-02-26 10:00:00 +0900
categories: [architecture]
tags: [アーキテクチャ, LLM, システム設計, 本番運用, 信頼性]
author: "AI Native Engineer"
reading_time: 16
---

## LLMをシステムに組み込む難しさ

LLMをプロダクションに組み込む際、従来のAPIとは異なる特性に直面します。

| 従来のAPI | LLM |
|---|---|
| 決定論的な出力 | 確率的・非決定論的な出力 |
| ms〜秒オーダーのレイテンシ | 秒〜10秒オーダーのレイテンシ |
| 安定したコスト | トークン数に依存する変動コスト |
| 明確なエラーコード | 品質の劣化は「エラー」にならない |
| バージョン管理が容易 | モデルアップデートで動作が変わる |

これらの特性を考慮した設計が必要です。

## パターン1: Gatewayパターン

LLMへのアクセスを集約するゲートウェイを設けます。

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class LLMRequest:
    prompt: str
    model: str = "gpt-4o-mini"
    max_tokens: int = 1000
    temperature: float = 0.0
    user_id: Optional[str] = None

@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    cached: bool = False

class LLMGateway:
    """LLMへのアクセスを集約するゲートウェイ"""
    
    def __init__(self, cache=None, rate_limiter=None):
        self.cache = cache
        self.rate_limiter = rate_limiter
        self._clients = {}
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        # レート制限チェック
        if self.rate_limiter:
            await self.rate_limiter.acquire(request.user_id)
        
        # キャッシュチェック
        if self.cache:
            cached = await self.cache.get(request)
            if cached:
                logger.info(f"Cache hit for user {request.user_id}")
                return cached
        
        # LLM呼び出し
        start_time = time.time()
        try:
            response = await self._call_llm(request)
            latency_ms = (time.time() - start_time) * 1000
            
            result = LLMResponse(
                content=response.content,
                model=request.model,
                tokens_used=response.usage.total_tokens,
                latency_ms=latency_ms,
            )
            
            # キャッシュ保存
            if self.cache:
                await self.cache.set(request, result)
            
            # メトリクス記録
            self._record_metrics(request, result)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM error: {e}", exc_info=True)
            raise
    
    def _record_metrics(self, request: LLMRequest, response: LLMResponse):
        # メトリクスをPrometheus/Datadogなどに送信
        logger.info(
            "llm_request",
            extra={
                "model": response.model,
                "tokens": response.tokens_used,
                "latency_ms": response.latency_ms,
                "user_id": request.user_id,
            }
        )
```

## パターン2: セマンティックキャッシュ

完全一致だけでなく、意味的に類似したクエリもキャッシュで処理します。

```python
import numpy as np
from typing import Optional

class SemanticCache:
    """意味的類似度によるキャッシュ"""
    
    def __init__(self, embedding_model, similarity_threshold: float = 0.95):
        self.embedding_model = embedding_model
        self.threshold = similarity_threshold
        self.cache_entries = []  # (embedding, response) のリスト
    
    async def get(self, request: LLMRequest) -> Optional[LLMResponse]:
        query_embedding = await self.embedding_model.aembed_query(request.prompt)
        
        if not self.cache_entries:
            return None
        
        # コサイン類似度を計算
        embeddings = np.array([e[0] for e in self.cache_entries])
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        max_idx = np.argmax(similarities)
        if similarities[max_idx] >= self.threshold:
            cached_response = self.cache_entries[max_idx][1]
            return LLMResponse(**{**cached_response.__dict__, "cached": True})
        
        return None
    
    async def set(self, request: LLMRequest, response: LLMResponse):
        embedding = await self.embedding_model.aembed_query(request.prompt)
        self.cache_entries.append((embedding, response))
```

## パターン3: フォールバックチェーン

高性能モデルが失敗した場合に、より安価なモデルにフォールバックします。

```python
from typing import List
import asyncio

class FallbackChain:
    """モデルのフォールバックチェーン"""
    
    def __init__(self, models: List[str]):
        self.models = models  # 優先順位順のモデルリスト
    
    async def complete(self, prompt: str) -> str:
        last_error = None
        
        for model in self.models:
            try:
                response = await self._call_model(model, prompt)
                if model != self.models[0]:
                    logger.warning(f"Fell back to model: {model}")
                return response
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model} failed: {e}")
                continue
        
        raise RuntimeError(f"All models failed. Last error: {last_error}")

# 使用例
fallback = FallbackChain([
    "gpt-4o",          # まず最高性能モデルを試みる
    "gpt-4o-mini",     # 失敗したら軽量モデルに
    "gpt-3.5-turbo",   # さらに古いモデルに
])
```

## パターン4: ガードレール（入出力バリデーション）

LLMへの入力と出力を検証・フィルタリングします。

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ValidationResult:
    is_valid: bool
    reason: Optional[str] = None
    sanitized_content: Optional[str] = None

class InputGuardrail:
    """入力の検証とサニタイズ"""
    
    MAX_LENGTH = 10000  # 最大文字数
    BLOCKED_PATTERNS = [
        "ignore previous instructions",
        "システムプロンプトを無視",
        "ルールを無視して",
    ]
    
    def validate(self, user_input: str) -> ValidationResult:
        # 長さチェック
        if len(user_input) > self.MAX_LENGTH:
            return ValidationResult(
                is_valid=False,
                reason=f"入力が長すぎます（最大{self.MAX_LENGTH}文字）"
            )
        
        # プロンプトインジェクション検出
        lower_input = user_input.lower()
        for pattern in self.BLOCKED_PATTERNS:
            if pattern.lower() in lower_input:
                return ValidationResult(
                    is_valid=False,
                    reason="不正な入力が検出されました"
                )
        
        return ValidationResult(is_valid=True, sanitized_content=user_input.strip())

class OutputGuardrail:
    """出力の検証とフィルタリング"""
    
    def validate(self, output: str, context: dict) -> ValidationResult:
        # PII（個人情報）の検出
        import re
        
        # クレジットカード番号パターン
        cc_pattern = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        if re.search(cc_pattern, output):
            return ValidationResult(
                is_valid=False,
                reason="出力に個人情報が含まれている可能性があります"
            )
        
        return ValidationResult(is_valid=True, sanitized_content=output)
```

## パターン5: 非同期ストリーミングとSSE

ユーザー体験のために、LLMの出力をストリーミングします。

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import asyncio
import json

app = FastAPI()
client = AsyncOpenAI()

@app.post("/chat/stream")
async def stream_chat(request: dict):
    user_message = request.get("message", "")
    
    async def generate():
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_message}],
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                data = json.dumps({"content": chunk.choices[0].delta.content})
                yield f"data: {data}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
```

## モニタリングとオブザーバビリティ

LLMシステムの監視には特別な指標が必要です。

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# メトリクス定義
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'status']
)

llm_latency_seconds = Histogram(
    'llm_latency_seconds',
    'LLM request latency',
    ['model'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

llm_tokens_used = Counter(
    'llm_tokens_used_total',
    'Total tokens consumed',
    ['model', 'token_type']  # prompt/completion
)

llm_cost_dollars = Counter(
    'llm_cost_dollars_total',
    'Estimated cost in dollars',
    ['model']
)

# コスト計算（モデルごとのトークン単価）
TOKEN_COSTS = {
    "gpt-4o": {"input": 0.0000025, "output": 0.00001},
    "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
}

def record_llm_metrics(model: str, prompt_tokens: int, completion_tokens: int, 
                        latency: float, success: bool):
    status = "success" if success else "error"
    llm_requests_total.labels(model=model, status=status).inc()
    llm_latency_seconds.labels(model=model).observe(latency)
    llm_tokens_used.labels(model=model, token_type="prompt").inc(prompt_tokens)
    llm_tokens_used.labels(model=model, token_type="completion").inc(completion_tokens)
    
    if model in TOKEN_COSTS:
        cost = (
            prompt_tokens * TOKEN_COSTS[model]["input"] +
            completion_tokens * TOKEN_COSTS[model]["output"]
        )
        llm_cost_dollars.labels(model=model).inc(cost)
```

## よくある落とし穴

### 1. コンテキストウィンドウの誤解

```python
# ❌ 悪い例：全てのメッセージ履歴を送り続ける
messages = conversation_history  # 無限に増え続ける

# ✅ 良い例：適切なウィンドウサイズを維持
MAX_HISTORY_TOKENS = 4000

def get_context_messages(history: list, max_tokens: int) -> list:
    """トークン予算内で最新のメッセージを返す"""
    result = []
    total_tokens = 0
    
    for msg in reversed(history):
        msg_tokens = estimate_tokens(msg["content"])
        if total_tokens + msg_tokens > max_tokens:
            break
        result.insert(0, msg)
        total_tokens += msg_tokens
    
    return result
```

### 2. エラーメッセージの漏洩

```python
# ❌ 悪い例：内部エラーをそのままユーザーに返す
try:
    response = llm.complete(prompt)
except Exception as e:
    return {"error": str(e)}  # APIキーや内部情報が漏洩する可能性

# ✅ 良い例：エラーを適切にマスク
try:
    response = llm.complete(prompt)
except RateLimitError:
    return {"error": "現在リクエストが集中しています。しばらくお待ちください。"}
except Exception as e:
    logger.error(f"LLM error: {e}", exc_info=True)  # 内部ログには詳細を記録
    return {"error": "処理中にエラーが発生しました。"}
```

### 3. 非決定性の扱い

```python
# テストでは temperature=0 にしてもわずかな非決定性がある
# Evalsフレームワークで品質を継続的に測定することが重要

def run_regression_test(test_cases: list, query_engine) -> dict:
    """回帰テスト: モデル更新後の品質を確認"""
    results = {"pass": 0, "fail": 0}
    
    for case in test_cases:
        response = query_engine.query(case["input"])
        if evaluate_response(str(response), case["expected_keywords"]):
            results["pass"] += 1
        else:
            results["fail"] += 1
            logger.warning(f"Test failed: {case['input'][:50]}")
    
    return results
```

## まとめ

LLMをプロダクションに組み込む際の重要なパターンをカバーしました：

1. **Gatewayパターン**でアクセスを集約・制御
2. **セマンティックキャッシュ**でコスト削減
3. **フォールバックチェーン**で信頼性向上
4. **ガードレール**で安全性確保
5. **ストリーミング**でUX改善
6. **オブザーバビリティ**でコストと品質を可視化

LLMは強力なツールですが、適切なアーキテクチャなしには本番環境で安定稼働させることは困難です。これらのパターンを参考に、堅牢なAIシステムを構築してください。
