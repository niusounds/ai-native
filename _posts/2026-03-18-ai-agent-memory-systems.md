---
layout: post
title: "AIエージェントのメモリシステム設計：短期・長期・エピソード記憶を使いこなす実装ガイド"
description: "AIエージェントに「記憶」を与える4種類のメモリシステムを解説。作業記憶・エピソード記憶・意味記憶・手続き記憶の設計パターンと、MemoryとRAGを組み合わせた実装例を紹介します。"
date: 2026-03-18 10:00:00 +0900
categories: [ai-agents]
tags: [メモリシステム, AIエージェント, RAG, ベクトルDB, LangChain, 長期記憶, 上級, エージェント設計]
author: "AI Native Engineer"
reading_time: 17
---

## はじめに：「なぜあなたは私のことを覚えていないのか？」

あなたが毎日使うAIアシスタントに、同じことを何度も説明しなければならない経験はありませんか？

「私はPythonをメインで使っています」「プロジェクトのコーディング規約はこれです」「いつも日本語で答えてください」——これらを毎回のセッションで説明するのは、きわめて非効率です。

人間のチームメンバーは時間とともに文脈を学習し、あなたの好み、過去の失敗、チームの慣習を記憶します。AIエージェントも同じことができるべきです。

2026年現在、**メモリシステムはAIエージェントのアーキテクチャで最も急速に進化している領域の一つ**です。単なる「会話履歴の保存」を超えて、人間の認知科学から着想を得た多層的なメモリ設計が普及しています。

この記事では、AIエージェントのメモリを以下の観点から体系的に解説します：

- 4種類のメモリタイプとその役割
- 各メモリの実装パターンとコード例
- メモリシステム全体のアーキテクチャ設計
- 実運用で注意すべき落とし穴

## メモリの4分類：認知科学からの借用

人間の記憶研究から、AIエージェントのメモリを4種類に分類するモデルが広く使われています。

| メモリタイプ | 人間の例 | AIエージェントでの例 |
|------------|---------|------------------|
| **作業記憶（Working Memory）** | 電話番号を一時的に覚える | 現在の会話コンテキスト・ツール実行の中間結果 |
| **エピソード記憶（Episodic Memory）** | 「先週の会議でこんな話をした」 | 過去のタスク実行ログ・ユーザーとのやり取り |
| **意味記憶（Semantic Memory）** | 「東京は日本の首都だ」という知識 | ドメイン知識・ユーザープロファイル・ファクト |
| **手続き記憶（Procedural Memory）** | 自転車の乗り方 | 成功した問題解決パターン・ワークフロー |

この4分類を念頭に置いて、それぞれの実装を見ていきましょう。

## 1. 作業記憶（Working Memory）：コンテキストウィンドウの賢い管理

### 概念

作業記憶は、エージェントが**現在のタスクを実行するために一時的に保持する情報**です。LLMにとっては、コンテキストウィンドウそのものが作業記憶に相当します。

現代のLLMはコンテキストウィンドウが大幅に拡大しましたが（200K〜1Mトークン）、無制限ではありません。作業記憶の設計では、**何を入れて、何を外すか**の戦略が重要です。

### 実装パターン：スライディングウィンドウ + 要約

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class WorkingMemoryManager:
    """
    スライディングウィンドウ方式の作業記憶管理。
    一定件数を超えた古いメッセージは要約して圧縮する。
    """

    def __init__(self, llm: ChatOpenAI, max_messages: int = 20):
        self.llm = llm
        self.max_messages = max_messages
        self.messages: list = []
        self.summary: str = ""

    def add_message(self, message):
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self._compress()

    def _compress(self):
        """古いメッセージを要約して圧縮する"""
        # 最新の半分は保持し、古い半分を要約
        split_point = len(self.messages) // 2
        old_messages = self.messages[:split_point]
        self.messages = self.messages[split_point:]

        # 既存の要約と古いメッセージをまとめて再要約
        history_text = "\n".join([
            f"{m.type}: {m.content}" for m in old_messages
        ])

        prompt = f"""以下の会話履歴を簡潔に要約してください。
重要な決定事項、ユーザーの好み、未解決の問題に焦点を当ててください。

既存の要約:
{self.summary}

新しい会話:
{history_text}

要約:"""

        response = self.llm.invoke(prompt)
        self.summary = response.content

    def get_context(self) -> list:
        """現在のコンテキストを返す（要約 + 最近のメッセージ）"""
        context = []
        if self.summary:
            context.append(SystemMessage(
                content=f"これまでの会話の要約:\n{self.summary}"
            ))
        context.extend(self.messages)
        return context
```

### ポイント：トークン数を意識した管理

```python
import tiktoken

def count_tokens(messages: list, model: str = "gpt-4o") -> int:
    """メッセージリストのトークン数を計算"""
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        # メッセージのオーバーヘッド（約4トークン/メッセージ）
        total += 4
        total += len(enc.encode(str(msg.content)))
    return total

# トークン数ベースの圧縮
MAX_TOKENS = 100_000  # コンテキストの上限の約半分を目安に
if count_tokens(messages) > MAX_TOKENS:
    manager._compress()
```

## 2. エピソード記憶（Episodic Memory）：過去の経験を蓄積する

### 概念

エピソード記憶は、**過去にエージェントが実行したタスクや体験した出来事**を保存します。「先週このユーザーから同じ質問が来た」「このAPIは503エラーを返しやすい」といった経験知が蓄積されます。

### 実装パターン：ベクトルDBによる意味的検索

エピソード記憶の実装には、ベクトルデータベース（ChromaDB, Qdrant, pgvector等）が最適です。過去の経験を埋め込みベクトルとして保存し、類似した状況を素早く検索できます。

```python
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import chromadb
from langchain_openai import OpenAIEmbeddings

@dataclass
class Episode:
    """1つのエピソード（過去の体験）"""
    task_description: str      # タスクの内容
    actions_taken: list[str]   # 実行したアクション
    outcome: str               # 結果（成功/失敗）
    lesson_learned: str        # 学んだこと
    timestamp: str             # 発生日時
    metadata: dict             # その他のメタデータ

class EpisodicMemory:
    """
    ベクトルDBを使ったエピソード記憶。
    過去の体験から関連する経験を検索する。
    """

    def __init__(self, collection_name: str = "agent_episodes"):
        self.client = chromadb.PersistentClient(path="./memory_store")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def store(self, episode: Episode) -> str:
        """エピソードを記憶に保存"""
        # タスクの説明とアクション、学びを結合してベクトル化
        text = f"""
タスク: {episode.task_description}
結果: {episode.outcome}
学び: {episode.lesson_learned}
"""
        embedding = self.embeddings.embed_query(text)
        episode_id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        self.collection.add(
            ids=[episode_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "task": episode.task_description,
                "outcome": episode.outcome,
                "lesson": episode.lesson_learned,
                "timestamp": episode.timestamp,
                "actions": json.dumps(episode.actions_taken, ensure_ascii=False),
            }]
        )
        return episode_id

    def recall(self, current_situation: str, n_results: int = 3) -> list[dict]:
        """現在の状況に関連する過去のエピソードを検索"""
        embedding = self.embeddings.embed_query(current_situation)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        episodes = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            episodes.append({
                "relevance_score": 1 - dist,  # cosine距離を類似度に変換
                "task": meta["task"],
                "outcome": meta["outcome"],
                "lesson": meta["lesson"],
                "actions": json.loads(meta["actions"]),
                "timestamp": meta["timestamp"],
            })

        return episodes

    def format_for_prompt(self, situation: str) -> str:
        """関連するエピソードをプロンプト用テキストにフォーマット"""
        episodes = self.recall(situation)
        if not episodes:
            return ""

        lines = ["【関連する過去の経験】"]
        for ep in episodes:
            lines.append(f"- タスク: {ep['task']}")
            lines.append(f"  結果: {ep['outcome']}")
            lines.append(f"  教訓: {ep['lesson']}")
            lines.append("")

        return "\n".join(lines)
```

### 活用例：タスク開始前に関連する過去経験を注入

```python
# エージェントがタスクを開始するときに過去の経験を参照
async def execute_task(agent, task: str, memory: EpisodicMemory):
    # 関連するエピソードを取得
    past_experience = memory.format_for_prompt(task)

    system_prompt = f"""あなたは経験豊富なAIアシスタントです。

{past_experience}

上記の過去の経験を参考に、同じミスを避けながらタスクを実行してください。"""

    result = await agent.run(task, system_prompt=system_prompt)

    # タスク完了後にエピソードを記録
    episode = Episode(
        task_description=task,
        actions_taken=result.actions,
        outcome="成功" if result.success else "失敗",
        lesson_learned=result.lesson if hasattr(result, 'lesson') else "",
        timestamp=datetime.now().isoformat(),
        metadata={"duration_ms": result.duration_ms}
    )
    memory.store(episode)

    return result
```

## 3. 意味記憶（Semantic Memory）：構造化された知識の管理

### 概念

意味記憶は、**エージェントが持つ静的・準静的な知識**です。ユーザープロファイル、ドメイン知識、設定情報、ファクトベースが該当します。

エピソード記憶（出来事ベース）と異なり、意味記憶は**「Aはこういうものだ」という概念や関係性**を扱います。

### 実装パターン：ユーザープロファイルの自動構築

```python
from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

class UserProfile(BaseModel):
    """ユーザーについての意味記憶"""
    name: Optional[str] = None
    preferred_language: str = "日本語"
    technical_level: str = Field(
        default="intermediate",
        description="beginner / intermediate / expert"
    )
    primary_languages: list[str] = Field(default_factory=list)
    coding_style_preferences: list[str] = Field(default_factory=list)
    project_context: Optional[str] = None
    communication_preferences: list[str] = Field(default_factory=list)
    last_updated: Optional[str] = None

class SemanticMemory:
    """
    ユーザープロファイルとドメイン知識の管理。
    会話から自動的にプロファイルを更新する。
    """

    def __init__(self, llm: ChatOpenAI, user_id: str):
        self.llm = llm
        self.user_id = user_id
        self.profile = self._load_profile()
        self.parser = PydanticOutputParser(pydantic_object=UserProfile)

    def _load_profile(self) -> UserProfile:
        """保存済みプロファイルを読み込む（DBやファイルから）"""
        # 実装例ではJSONファイルを使用
        import os, json
        profile_path = f"./profiles/{self.user_id}.json"
        if os.path.exists(profile_path):
            with open(profile_path) as f:
                return UserProfile(**json.load(f))
        return UserProfile()

    def save_profile(self):
        """プロファイルを永続化する"""
        import os, json
        os.makedirs("./profiles", exist_ok=True)
        with open(f"./profiles/{self.user_id}.json", "w") as f:
            json.dump(self.profile.model_dump(), f, ensure_ascii=False, indent=2)

    def extract_and_update(self, conversation: str):
        """
        会話テキストからユーザー情報を抽出してプロファイルを更新。
        LLMを使って自動的に情報を引き出す。
        """
        prompt = f"""以下の会話から、ユーザーについての情報を抽出してください。

現在のプロファイル:
{self.profile.model_dump_json(indent=2)}

新しい会話:
{conversation}

会話から読み取れる情報でプロファイルを更新してください。
新しい情報がない項目は現在の値を維持してください。

{self.parser.get_format_instructions()}"""

        response = self.llm.invoke(prompt)
        try:
            updated_profile = self.parser.parse(response.content)
            updated_profile.last_updated = datetime.now().isoformat()
            self.profile = updated_profile
            self.save_profile()
        except Exception as e:
            # パース失敗時は現在のプロファイルを維持
            print(f"プロファイル更新スキップ: {e}")

    def get_system_context(self) -> str:
        """エージェントのシステムプロンプトに追加するユーザーコンテキスト"""
        p = self.profile
        lines = ["【ユーザー情報】"]
        if p.name:
            lines.append(f"- 名前: {p.name}")
        lines.append(f"- 技術レベル: {p.technical_level}")
        if p.primary_languages:
            lines.append(f"- 使用言語: {', '.join(p.primary_languages)}")
        if p.coding_style_preferences:
            lines.append(f"- コーディングスタイル: {', '.join(p.coding_style_preferences)}")
        if p.project_context:
            lines.append(f"- プロジェクト背景: {p.project_context}")
        return "\n".join(lines)
```

## 4. 手続き記憶（Procedural Memory）：成功パターンの自動学習

### 概念

手続き記憶は、エージェントが**「どうやってやるか」を覚える**メモリです。人間が自転車の乗り方を一度覚えると意識せずに実行できるように、エージェントも成功した問題解決のパターンを手続きとして記憶します。

これは現在最もフロンティアに近いメモリタイプで、**エージェントが自律的に自分自身のツールやワークフローを改善していく**基盤となります。

### 実装パターン：ワークフローテンプレートの蓄積

```python
from dataclasses import dataclass

@dataclass
class Procedure:
    """再利用可能な手続きテンプレート"""
    name: str
    trigger_pattern: str    # どんな状況でこの手続きを使うか
    steps: list[str]        # 実行ステップ
    success_rate: float     # 成功率（0.0〜1.0）
    usage_count: int        # 使用回数
    last_updated: str

class ProceduralMemory:
    """
    成功したワークフローを手続きとして蓄積・改善する。
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        # エピソード記憶と同様にベクトルDBで管理
        self.client = chromadb.PersistentClient(path="./memory_store")
        self.collection = self.client.get_or_create_collection("procedures")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def find_applicable_procedure(self, task: str) -> Optional[Procedure]:
        """タスクに適用できる手続きを検索"""
        embedding = self.embeddings.embed_query(task)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["metadatas", "distances"]
        )
        if not results["ids"][0]:
            return None

        distance = results["distances"][0][0]
        # 類似度が高い（距離が小さい）かつ成功率が高い場合のみ採用
        meta = results["metadatas"][0][0]
        if distance < 0.3 and float(meta["success_rate"]) > 0.7:
            return Procedure(
                name=meta["name"],
                trigger_pattern=meta["trigger_pattern"],
                steps=json.loads(meta["steps"]),
                success_rate=float(meta["success_rate"]),
                usage_count=int(meta["usage_count"]),
                last_updated=meta["last_updated"]
            )
        return None

    def record_execution(self, procedure_name: str, success: bool):
        """手続きの実行結果を記録し成功率を更新"""
        # 実装省略：成功率の移動平均を更新
        pass

{% raw %}
    def synthesize_new_procedure(
        self,
        task: str,
        successful_episodes: list[dict]
    ) -> Optional[Procedure]:
        """
        複数の成功エピソードから新しい汎用手続きを合成。
        LLMがパターンを抽出して再利用可能な手順を生成する。
        """
        if len(successful_episodes) < 3:
            return None  # データ不足

        episodes_text = "\n\n".join([
            f"エピソード{i+1}:\n"
            f"タスク: {ep['task']}\n"
            f"実行ステップ: {', '.join(ep['actions'])}\n"
            f"教訓: {ep['lesson']}"
            for i, ep in enumerate(successful_episodes)
        ])

        prompt = f"""以下の成功した実行事例から、共通のパターンを抽出して、
再利用可能な手続きテンプレートを作成してください。

タスク種別: {task}

成功事例:
{episodes_text}

以下のJSON形式で回答してください:
{{
    "name": "手続きの名前",
    "trigger_pattern": "この手続きを適用すべき状況の説明",
    "steps": ["ステップ1", "ステップ2", "ステップ3"]
}}"""

        response = self.llm.invoke(prompt)
        # JSONパースして手続きを保存
        # ... (実装省略)
{% endraw %}

```

## メモリシステムの統合アーキテクチャ

4種類のメモリを統合して、一貫したインターフェースで提供するオーケストレーターを作りましょう。

```python
class AgentMemoryOrchestrator:
    """
    4種類のメモリを統合管理するオーケストレーター。
    エージェントはこのクラスを通じてすべての記憶にアクセスする。
    """

    def __init__(self, llm: ChatOpenAI, user_id: str):
        self.working = WorkingMemoryManager(llm)
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory(llm, user_id)
        self.procedural = ProceduralMemory(llm)

    def build_system_prompt(self, current_task: str) -> str:
        """現在のタスクに最適なシステムプロンプトを構築"""
        parts = ["あなたは経験豊富なAIアシスタントです。\n"]

        # 意味記憶：ユーザーコンテキスト（常に含める）
        user_context = self.semantic.get_system_context()
        if user_context:
            parts.append(user_context)

        # エピソード記憶：関連する過去の経験
        past_exp = self.episodic.format_for_prompt(current_task)
        if past_exp:
            parts.append(past_exp)

        # 手続き記憶：適用可能なワークフロー
        procedure = self.procedural.find_applicable_procedure(current_task)
        if procedure:
            steps_text = "\n".join(
                f"  {i+1}. {step}"
                for i, step in enumerate(procedure.steps)
            )
            parts.append(
                f"【推奨ワークフロー: {procedure.name}】\n"
                f"（成功率: {procedure.success_rate:.0%}）\n"
                f"{steps_text}"
            )

        return "\n\n".join(parts)

    def get_messages_with_context(self) -> list:
        """作業記憶のメッセージ（要約コンテキスト付き）を取得"""
        return self.working.get_context()

    def after_task(
        self,
        task: str,
        actions: list[str],
        success: bool,
        lesson: str = ""
    ):
        """タスク完了後の記憶更新"""
        # エピソード記憶に記録
        episode = Episode(
            task_description=task,
            actions_taken=actions,
            outcome="成功" if success else "失敗",
            lesson_learned=lesson,
            timestamp=datetime.now().isoformat(),
            metadata={}
        )
        self.episodic.store(episode)

        # 意味記憶：会話からプロファイルを更新
        conversation = f"タスク: {task}\n結果: {episode.outcome}"
        self.semantic.extract_and_update(conversation)
```

## 実装時の注意点とベストプラクティス

### 1. メモリの鮮度管理：古い記憶は重みを下げる

エピソード記憶は時間とともに陳腐化します。古いエピソードには低いスコアを与えましょう。

```python
from datetime import datetime, timedelta

def apply_temporal_decay(episodes: list[dict], decay_days: int = 30) -> list[dict]:
    """時間減衰を適用して古いエピソードのスコアを下げる"""
    now = datetime.now()
    for ep in episodes:
        ep_time = datetime.fromisoformat(ep["timestamp"])
        days_old = (now - ep_time).days
        # 指数的減衰：30日で約37%に
        decay = 0.5 ** (days_old / decay_days)
        ep["relevance_score"] *= decay
    return sorted(episodes, key=lambda x: x["relevance_score"], reverse=True)
```

### 2. プライバシーとセキュリティ：何を記憶させるか

すべての会話を記録するべきではありません：

- ✅ ユーザーの技術的好み、コーディングスタイル
- ✅ プロジェクトの文脈、チームの規約
- ✅ よくある質問パターンと解決策
- ❌ 認証情報、APIキー、パスワード
- ❌ 個人の機密情報（医療・金融等）
- ❌ 一時的なデバッグ情報

```python
import re

SENSITIVE_PATTERNS = [
    r'(password|passwd|secret|api[_-]?key|token)\s*[=:]\s*\S+',
    r'[A-Za-z0-9+/]{40,}={0,2}',  # Base64エンコードされたシークレット
    r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # クレジットカード番号
]

def sanitize_before_storing(text: str) -> str:
    """機密情報をマスクしてから記憶に保存する"""
    for pattern in SENSITIVE_PATTERNS:
        text = re.sub(pattern, '[REDACTED]', text, flags=re.IGNORECASE)
    return text
```

### 3. メモリのサイズと検索コストのバランス

ベクトルDBの検索はスケールするにつれてコストがかかります。実用的な制限を設けましょう：

```python
# エピソード記憶の上限管理
MAX_EPISODES = 10_000

def prune_old_episodes(memory: EpisodicMemory):
    """
    エピソード数が上限を超えたら、
    古くて使用頻度の低いエピソードを削除する。
    """
    count = memory.collection.count()
    if count > MAX_EPISODES:
        # 古いエピソードを一括削除（実装はDBに依存）
        # 例：タイムスタンプでソートして古いものから削除
        pass
```

### 4. LangGraph との統合

最新のエージェントフレームワークでは、メモリを State として管理する設計が主流です：

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    current_task: str
    memory_context: str     # メモリから注入されたコンテキスト
    actions_taken: list[str]
    task_complete: bool

def memory_injection_node(state: AgentState) -> AgentState:
    """グラフの最初のノードでメモリコンテキストを注入"""
    memory = get_memory_orchestrator(state["user_id"])
    context = memory.build_system_prompt(state["current_task"])
    return {"memory_context": context}

def memory_update_node(state: AgentState) -> AgentState:
    """タスク完了後にメモリを更新するノード"""
    memory = get_memory_orchestrator(state["user_id"])
    memory.after_task(
        task=state["current_task"],
        actions=state["actions_taken"],
        success=state["task_complete"]
    )
    return {}

# グラフに組み込む
builder = StateGraph(AgentState)
builder.add_node("inject_memory", memory_injection_node)
builder.add_node("agent", agent_node)
builder.add_node("update_memory", memory_update_node)

builder.set_entry_point("inject_memory")
builder.add_edge("inject_memory", "agent")
builder.add_edge("agent", "update_memory")
builder.add_edge("update_memory", END)
```

## まとめ：記憶を持つエージェントの設計原則

AIエージェントのメモリシステムを設計する際の核心的な原則をまとめます：

| 原則 | 内容 |
|-----|------|
| **分離の原則** | 4種類のメモリを明確に分離して設計する |
| **漸進的改善** | まずエピソード記憶だけ実装し、徐々に追加する |
| **プライバシーファースト** | 記憶する前に必ず機密情報をフィルタリング |
| **鮮度管理** | 古い記憶は自動的に重みを下げるか削除する |
| **観測可能性** | メモリの読み書きをログに残しデバッグしやすくする |

### 今すぐ実装できるクイックスタート

まずは「意味記憶」だけをシンプルに実装するところから始めましょう：

```python
# 最小限の実装：ユーザープロファイルをJSONで管理するだけ
import json, os

def load_user_memory(user_id: str) -> dict:
    path = f"./memory/{user_id}.json"
    return json.load(open(path)) if os.path.exists(path) else {}

def save_user_memory(user_id: str, data: dict):
    os.makedirs("./memory", exist_ok=True)
    json.dump(data, open(f"./memory/{user_id}.json", "w"), ensure_ascii=False)

# これだけでも「ユーザーの設定を覚える」エージェントが作れます
```

## 参考資料

- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
- [LangGraph Memory Documentation](https://langchain-ai.github.io/langgraph/concepts/memory/)
- [Cognitive Architectures for Language Agents (CoALA)](https://arxiv.org/abs/2309.02427)
- [Zep AI - Production Memory Layer](https://github.com/getzep/zep)
- [mem0 - The Memory Layer for AI Apps](https://github.com/mem-labs/mem0)

---

**関連記事:**
- [コンテキストエンジニアリング：LLMのパフォーマンスを最大化するコンテキスト設計術](/ai-agents/2026/03/13/context-engineering-guide.html)
- [マルチエージェントシステム設計パターン](/ai-agents/2026/03/17/multi-agent-system-design-patterns.html)
- [AIコーディングエージェント完全活用ガイド](/ai-agents/2026/03/15/ai-coding-agents-guide.html)
