---
layout: post
title: "コンテキストエンジニアリング：LLMのパフォーマンスを最大化するコンテキスト設計術"
description: "プロンプト設計を超えた次のステージ「コンテキストエンジニアリング」を解説。LLMが受け取るすべての情報を設計し、精度・コスト・速度を最適化する実践的なテクニックを紹介します。"
date: 2026-03-13 10:00:00 +0900
categories: [prompt-engineering]
tags: [コンテキストエンジニアリング, LLM, プロンプト, 上級, 最適化, コスト削減]
author: "AI Native Engineer"
reading_time: 14
---

## はじめに：プロンプトエンジニアリングの次のステージ

「良いプロンプトを書く」というスキルは、もはや入門レベルの話になりました。GPT-4oやClaude 3.7 Sonnet、Gemini 2.0 Flashといったフロンティアモデルが普及した今、AIネイティブエンジニアが次に習得すべきスキルは **コンテキストエンジニアリング（Context Engineering）** です。

コンテキストエンジニアリングとは、**LLMが1回の推論で受け取るすべての情報（コンテキストウィンドウ）を、意図的かつ体系的に設計すること**です。単にシステムプロンプトを上手く書くだけでなく、以下のすべてを統合的に管理します：

- システムプロンプト
- 会話履歴
- RAGで取得したドキュメント
- ツール定義とその実行結果
- 数発のFew-shot例
- ユーザーの最新の入力

これらをどう組み合わせ、何を入れて何を捨てるかを設計する技術が、コンテキストエンジニアリングです。

この記事では、AIを使ったアプリケーション開発に携わる中級〜上級エンジニア向けに、コンテキストエンジニアリングの理論と実践テクニックを解説します。

---

## コンテキストウィンドウとは何か（復習）

LLMは1回の推論で処理できる情報量に上限があります。この上限を**コンテキストウィンドウ**と呼び、単位はトークン（おおよそ1単語〜数文字）です。

| モデル | コンテキストウィンドウ |
|--------|----------------------|
| GPT-4o | 128K トークン |
| Claude 3.7 Sonnet | 200K トークン |
| Gemini 1.5 Pro | 1M トークン |
| Llama 3.3 70B | 128K トークン |

200Kトークンは英語で約150,000ワード、日本語で約10万文字以上に相当します。「もう何でも入るじゃないか」と思うかもしれませんが、現実には以下の理由で**コンテキスト設計が重要**です：

1. **コスト**: 入力トークン数に応じて課金される。無駄な情報を詰め込むと費用がかさむ
2. **速度**: トークン数が増えると推論時間（Time To First Token）も増加する
3. **Lost in the Middle問題**: コンテキストが長くなるほど、中間部分の情報を見落としやすくなる
4. **ノイズ**: 無関係な情報が多いと、関連情報を見つける精度が落ちる

---

## コンテキストエンジニアリングの5つの原則

### 原則1：関連性の最大化（Signal-to-Noise Ratio）

コンテキストに含める情報は「多ければ良い」ではありません。**タスクに直接関係する情報の割合（Signal）を最大化し、ノイズを排除すること**が基本です。

**悪い例：** ドキュメント全体をRAGで取得して詰め込む

```python
# ❌ ノイズが多い
context = full_document  # 10,000 トークンのドキュメント全体

prompt = f"""
以下のドキュメントに基づいて質問に答えてください。

{context}

質問: {user_question}
"""
```

**良い例：** 関連チャンクのみを精選する

```python
# ✅ 関連情報だけを抽出
from openai import OpenAI

client = OpenAI()

def get_relevant_chunks(question: str, documents: list[str], top_k: int = 3) -> list[str]:
    """ベクトル検索で関連チャンクのみを取得"""
    # ベクトル類似度でtop_kチャンクを取得（vector_storeは事前構築済み）
    results = vector_store.similarity_search(question, k=top_k)
    return [r.page_content for r in results]

relevant_chunks = get_relevant_chunks(user_question, documents)
context = "\n\n---\n\n".join(relevant_chunks)  # 〜1,500 トークン程度

prompt = f"""
以下の参考情報に基づいて質問に答えてください。

参考情報:
{context}

質問: {user_question}
"""
```

### 原則2：情報の配置（Placement Matters）

LLMはコンテキストの**先頭と末尾の情報を最も良く記憶します**（Primary-Recency効果）。これを意識した情報配置が重要です。

```
[推奨される配置順序]

1. システムプロンプト（最優先事項・ペルソナ・制約）← 先頭
2. Few-shot例（あれば）
3. 参照ドキュメント・RAG結果（中間に配置）
4. 会話履歴（直近のもの）
5. ユーザーの現在の質問・タスク ← 末尾（最重要）
```

特に**ユーザーの質問は必ず末尾に置く**べきです。長い参照ドキュメントの後にユーザーの意図を置くことで、モデルはドキュメントを読みながら「何のために読んでいるか」を意識できます。

### 原則3：会話履歴の圧縮（Memory Compression）

チャットアプリケーションでは会話が長くなるほどコンテキストが肥大化します。**古い会話は要約して圧縮**することで、コストと速度を維持しながら文脈を保持できます。

```python
from anthropic import Anthropic

client = Anthropic()

class ConversationManager:
    def __init__(self, max_tokens: int = 4000, summary_threshold: int = 3000):
        self.messages = []
        self.summary = ""
        self.max_tokens = max_tokens
        self.summary_threshold = summary_threshold

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        
        # トークン数が閾値を超えたら古い会話を要約
        if self._estimate_tokens() > self.summary_threshold:
            self._compress_history()

    def _estimate_tokens(self) -> int:
        """簡易的なトークン数推定（1文字≈0.7トークンで計算）"""
        total_chars = sum(len(m["content"]) for m in self.messages)
        return int(total_chars * 0.7)

    def _compress_history(self):
        """古い会話履歴を要約して圧縮する"""
        # 直近3ターンは保持、それ以前を要約
        recent_messages = self.messages[-6:]  # 直近3ターン（user+assistant×3）
        old_messages = self.messages[:-6]
        
        if not old_messages:
            return
        
        # LLMを使って古い会話を要約
        summary_prompt = "以下の会話を3〜5文で要約してください：\n\n"
        for msg in old_messages:
            summary_prompt += f"{msg['role']}: {msg['content']}\n"
        
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",  # 要約には軽量モデルを使用
            max_tokens=500,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        new_summary = response.content[0].text
        # 既存のサマリーがあれば統合
        self.summary = f"{self.summary}\n{new_summary}".strip() if self.summary else new_summary
        self.messages = recent_messages

    def get_context_messages(self) -> list[dict]:
        """システム側に渡すメッセージリストを構築"""
        if not self.summary:
            return self.messages
        
        # 要約を先頭メッセージとして挿入
        summary_message = {
            "role": "user",
            "content": f"[会話の要約]\n{self.summary}\n\n[以下から現在の会話]"
        }
        return [summary_message, {"role": "assistant", "content": "承知しました。"}, *self.messages]
```

### 原則4：Few-shot例の戦略的活用

Few-shot例はモデルの出力フォーマットや思考パターンを誘導する強力な手段ですが、**例が多すぎるとトークンを大量消費**します。

以下の指針に従って Few-shot 例を使いましょう：

| 状況 | 推奨アプローチ |
|------|---------------|
| 出力フォーマットの指定 | 1〜2例で十分（多くても3例） |
| 複雑な推論タスク | 3〜5例（Chain-of-Thought形式で） |
| 分類タスク | クラスごとに1例（最大5例程度） |
| 一般的な質問応答 | Few-shot不要（指示のみで対応可） |

```python
# ✅ Few-shot例をタスクに応じて動的に選択する
def select_few_shot_examples(task_type: str, user_input: str) -> str:
    """
    タスクタイプとユーザー入力の類似度に基づいてFew-shot例を動的選択
    例のライブラリから最も関連性の高いものを選ぶ
    """
    example_library = {
        "code_review": [...],  # コードレビュー用の例
        "summarization": [...],  # 要約用の例
        "classification": [...],  # 分類用の例
    }
    
    # ベクトル検索で入力に最も近い例を2〜3個選択
    relevant_examples = vector_store.similarity_search(
        query=user_input,
        filter={"task_type": task_type},
        k=2
    )
    
    return format_examples(relevant_examples)
```

### 原則5：ツール定義の最適化

Function CallingやTool Useでは、**ツール定義自体もトークンを消費します**。ツールが増えるほどコンテキストが膨らみ、モデルが適切なツールを選択する精度も低下します。

```python
# ❌ 全ツールを常に渡す（50個のツールがある場合、定義だけで数千トークン）
all_tools = load_all_tools()  # 50個のツール
response = client.chat.completions.create(
    model="gpt-4o",
    tools=all_tools,  # 常に全ツールを渡す
    messages=messages
)

# ✅ ユーザーのインテントに基づいて関連ツールだけを動的に選択
def select_relevant_tools(user_message: str, all_tools: list) -> list:
    """インテント分類でタスクに必要なツールだけを抽出"""
    # 簡易的なキーワードベースの選択（実際はより高度な分類を推奨）
    intent_keywords = {
        "search": ["検索", "調べ", "探し"],
        "calculate": ["計算", "合計", "平均"],
        "file": ["ファイル", "保存", "読み込み"],
        "email": ["メール", "送信", "宛先"],
    }
    
    needed_categories = set()
    for category, keywords in intent_keywords.items():
        if any(kw in user_message for kw in keywords):
            needed_categories.add(category)
    
    # 関連カテゴリのツールだけを返す（最大10個程度に絞る）
    return [t for t in all_tools if t.get("category") in needed_categories][:10]
```

---

## 実践：コンテキストエンジニアリングパイプラインの構築

以上の原則を統合した、プロダクション対応のコンテキスト構築クラスを実装してみましょう。

```python
from dataclasses import dataclass, field
from typing import Any
import tiktoken  # OpenAI公式のトークンカウンター

@dataclass
class ContextBuilder:
    """
    LLMに渡すコンテキストを体系的に構築・管理するクラス
    """
    model: str = "gpt-4o"
    max_context_tokens: int = 100_000  # 安全マージンを持たせた上限
    
    system_prompt: str = ""
    few_shot_examples: list[dict] = field(default_factory=list)
    retrieved_documents: list[str] = field(default_factory=list)
    conversation_history: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    
    def _count_tokens(self, text: str) -> int:
        """テキストのトークン数をカウント"""
        enc = tiktoken.encoding_for_model(self.model)
        return len(enc.encode(text))
    
    def _count_messages_tokens(self, messages: list[dict]) -> int:
        """メッセージリスト全体のトークン数を推定"""
        total = 0
        for msg in messages:
            total += self._count_tokens(msg.get("content", ""))
            total += 4  # ロールや区切りのオーバーヘッド
        return total
    
    def add_rag_documents(self, docs: list[str], budget_tokens: int = 20_000):
        """
        RAG結果を追加。トークン予算内に収まるよう自動的にトリミング。
        """
        added_tokens = 0
        self.retrieved_documents = []
        
        for doc in docs:
            doc_tokens = self._count_tokens(doc)
            if added_tokens + doc_tokens > budget_tokens:
                # 予算オーバーの場合は残りを切り捨て
                break
            self.retrieved_documents.append(doc)
            added_tokens += doc_tokens
    
    def build_messages(self, user_input: str) -> list[dict]:
        """
        最終的なメッセージリストを構築。
        配置順序: システムプロンプト → Few-shot → RAG → 会話履歴 → ユーザー入力
        """
        messages = []
        
        # 1. システムプロンプト
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        # 2. Few-shot例（あれば）
        messages.extend(self.few_shot_examples)
        
        # 3. RAG取得ドキュメント（ユーザーメッセージとして埋め込む）
        if self.retrieved_documents:
            rag_content = "# 参考情報\n\n" + "\n\n---\n\n".join(self.retrieved_documents)
            messages.append({
                "role": "user",
                "content": rag_content
            })
            messages.append({
                "role": "assistant",
                "content": "参考情報を確認しました。ご質問にお答えします。"
            })
        
        # 4. 会話履歴
        messages.extend(self.conversation_history)
        
        # 5. 現在のユーザー入力（必ず末尾）
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        # トークン数チェック
        total_tokens = self._count_messages_tokens(messages)
        if total_tokens > self.max_context_tokens:
            raise ValueError(
                f"コンテキストがトークン上限を超えています: {total_tokens} > {self.max_context_tokens}\n"
                f"会話履歴の圧縮またはRAGドキュメント削減を検討してください。"
            )
        
        return messages
    
    def get_token_breakdown(self, user_input: str) -> dict[str, int]:
        """コンテキスト内のトークン消費内訳を可視化するデバッグ用メソッド"""
        return {
            "system_prompt": self._count_tokens(self.system_prompt),
            "few_shot_examples": self._count_messages_tokens(self.few_shot_examples),
            "rag_documents": sum(self._count_tokens(d) for d in self.retrieved_documents),
            "conversation_history": self._count_messages_tokens(self.conversation_history),
            "user_input": self._count_tokens(user_input),
            "tools": self._count_tokens(str(self.tools)),
        }


# 使用例
builder = ContextBuilder(
    model="gpt-4o",
    system_prompt="""あなたは社内のドキュメントアシスタントです。
与えられた参考情報のみに基づいて回答し、情報がない場合は「分かりません」と答えてください。""",
    max_context_tokens=80_000
)

# RAGで取得したドキュメントを追加（トークン予算20,000以内に自動調整）
builder.add_rag_documents(retrieved_docs, budget_tokens=20_000)

# 会話履歴を設定
builder.conversation_history = compressed_history

# デバッグ：トークン消費内訳を確認
breakdown = builder.get_token_breakdown(user_question)
print("トークン内訳:", breakdown)
# 出力例:
# トークン内訳: {'system_prompt': 87, 'few_shot_examples': 0, 
#                'rag_documents': 3420, 'conversation_history': 1250, 
#                'user_input': 45, 'tools': 0}

# 最終的なメッセージを構築
messages = builder.build_messages(user_question)
```

---

## コンテキストの品質を測定する：評価指標

コンテキストエンジニアリングの改善には、**定量的な評価**が不可欠です。以下の指標をモニタリングしましょう。

### 1. Context Relevance（文脈関連性）

RAGで取得したチャンクが実際に役立っているかを評価します。

```python
def evaluate_context_relevance(question: str, context_chunks: list[str]) -> float:
    """
    LLM-as-Judgeパターンで文脈の関連性を0〜1でスコアリング
    """
    evaluation_prompt = f"""
    質問に対して、以下の文脈チャンクがどの程度関連しているかを評価してください。
    
    質問: {question}
    
    文脈チャンク:
    {chr(10).join(f'{i+1}. {chunk[:200]}...' for i, chunk in enumerate(context_chunks))}
    
    0.0（全く関連なし）〜1.0（非常に関連あり）のスコアのみを数値で回答してください。
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": evaluation_prompt}],
        temperature=0,
        max_tokens=10
    )
    
    try:
        return float(response.choices[0].message.content.strip())
    except ValueError:
        return 0.0
```

### 2. Context Utilization（文脈活用率）

提供した文脈のうち、実際にモデルの回答に使われた割合を推定します。

| 指標 | 目標値 | 改善策 |
|------|--------|--------|
| Context Relevance | > 0.7 | チャンクサイズの調整、再ランキングの導入 |
| Answer Faithfulness | > 0.85 | RAGの精度向上、ハルシネーション対策プロンプト |
| Token Efficiency | < 0.3（入力/出力比） | 不要な文脈の削減、会話履歴の圧縮 |

---

## 高度なテクニック：プロンプトキャッシング

最後に、コスト最適化の切り札として**プロンプトキャッシング**を紹介します。

AnthropicのClaude APIとOpenAIのAPIは、コンテキストの特定部分をキャッシュする機能を提供しています。**変化しない部分（システムプロンプト、参照ドキュメント）をキャッシュすることで、入力コストを最大90%削減**できます。

```python
# Anthropic Claude でのプロンプトキャッシング
import anthropic

client = anthropic.Anthropic()

# 大きな参照ドキュメントをキャッシュ対象としてマーク
large_document = open("technical_specification.txt").read()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "あなたは技術仕様書のエキスパートアシスタントです。",
        },
        {
            "type": "text",
            "text": large_document,
            "cache_control": {"type": "ephemeral"}  # ← このブロックをキャッシュ
        }
    ],
    messages=[{"role": "user", "content": user_question}]
)

# cache_read_input_tokens が増えれば増えるほどコスト削減効果が高い
print(f"通常トークン: {response.usage.input_tokens}")
print(f"キャッシュ読込: {response.usage.cache_read_input_tokens}")
print(f"キャッシュ書込: {response.usage.cache_creation_input_tokens}")
```

キャッシングが特に効果的なシナリオ：

- **チャットボット**: システムプロンプト（数千トークン）を毎回キャッシュ
- **ドキュメントQ&A**: 同一ドキュメントへの複数の質問
- **コード解析**: 同一コードベースに対する繰り返しの分析タスク

---

## まとめ

コンテキストエンジニアリングの核心を振り返ります：

1. **Signal-to-Noise Ratio を最大化** — 関連情報だけを精選してコンテキストに入れる
2. **情報配置を最適化** — システムプロンプトを先頭に、ユーザー入力を末尾に
3. **会話履歴を圧縮** — 古い会話は要約してトークン消費を抑える
4. **Few-shot例を動的選択** — 全例を常に渡すのではなく、タスクに応じて精選する
5. **ツールを動的に絞り込む** — インテントに合ったツールのみを渡す
6. **プロンプトキャッシングを活用** — 変化しない大きな文脈をキャッシュしてコスト削減

「どのモデルを使うか」よりも「モデルに何を渡すか」を真剣に設計することが、AIネイティブエンジニアとして次のレベルに進む鍵です。コンテキストエンジニアリングをマスターすることで、同じモデルでも精度・コスト・速度の三拍子を大幅に改善できます。

---

## 参考資料

- [Anthropic Prompt Caching documentation](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)
- [Lost in the Middle: How Language Models Use Long Contexts (arXiv)](https://arxiv.org/abs/2307.03172)
- [RAGAS: Evaluation framework for RAG systems](https://github.com/explodinggradients/ragas)
- [tiktoken: OpenAI tokenizer](https://github.com/openai/tiktoken)
