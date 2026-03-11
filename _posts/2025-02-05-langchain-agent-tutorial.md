---
layout: post
title: "LangChainで作るAIエージェント入門：自律的に動作するボットを30分で構築する"
description: "LangChainを使って、Webを検索しコードを実行できるAIエージェントをゼロから構築します。ReActパターンの実装から本番対応のエラーハンドリングまでを解説。"
date: 2025-02-05 10:00:00 +0900
categories: [ai-agents]
tags: [LangChain, Python, エージェント, 実装, ReAct]
author: "AI Native Engineer"
reading_time: 15
---

## はじめに

「AIエージェント」という言葉は広く使われていますが、実際にどう実装するのか、具体的なコードを見たことがない方も多いはずです。

この記事では、**LangChainを使って実際に動くAIエージェントを30分で作る**ことを目標に、基礎から実装まで解説します。

## 事前準備

```bash
pip install langchain langchain-openai duckduckgo-search python-dotenv
```

```python
# .env
OPENAI_API_KEY=your_api_key_here
```

## 最小限のエージェント

まず最も単純なエージェントを作ってみましょう。

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import DuckDuckGoSearchRun
from langchain import hub

# LLMの設定
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ツールの設定
tools = [DuckDuckGoSearchRun()]

# ReActプロンプトを取得（HubからダウンロードOrローカルで定義）
prompt = hub.pull("hwchase17/react")

# エージェントの作成
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 実行
result = agent_executor.invoke({
    "input": "今日の東京の天気を調べて、明日の服装を提案してください"
})
print(result["output"])
```

`verbose=True` にすることで、エージェントがどのように思考・行動しているかを確認できます。

## ReActパターンの仕組み

エージェントの動作を理解するために、ReActパターンを詳しく見てみましょう。

```
入力: "今日の東京の天気を教えて"

ループ開始:
  [思考] まず東京の天気を検索する必要がある
  [行動] duckduckgo_search("東京 今日 天気 2025")
  [観察] "東京の本日の天気: 晴れ、最高気温28度、最低気温18度"
  
  [思考] 天気情報が得られた。これで回答できる
  [行動] Final Answer: "東京は晴れで最高気温28度です..."

ループ終了
```

## カスタムツールの作成

DuckDuckGo以外にも、独自のツールを簡単に追加できます。

```python
from langchain.tools import tool
from typing import Optional
import subprocess

@tool
def run_python_code(code: str) -> str:
    """
    Pythonコードを安全なサンドボックスで実行します。
    
    Args:
        code: 実行するPythonコード
        
    Returns:
        実行結果（stdout）またはエラーメッセージ
    """
    try:
        # 注意: 本番環境では必ずサンドボックス化すること
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=10  # タイムアウト設定
        )
        if result.returncode == 0:
            return result.stdout or "コードは正常に実行されました（出力なし）"
        else:
            return f"エラー: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "エラー: 実行タイムアウト（10秒）"

@tool
def read_file(filepath: str) -> str:
    """
    ファイルの内容を読み込みます。
    
    Args:
        filepath: 読み込むファイルのパス
        
    Returns:
        ファイルの内容
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"エラー: ファイル '{filepath}' が見つかりません"
    except Exception as e:
        return f"エラー: {str(e)}"

# ツールをエージェントに追加
tools = [
    DuckDuckGoSearchRun(),
    run_python_code,
    read_file,
]
```

## エラーハンドリングと本番対応

エージェントを本番で使うには、適切なエラーハンドリングが必要です。

```python
from langchain.agents import AgentExecutor
from langchain.callbacks.base import BaseCallbackHandler
import logging

logger = logging.getLogger(__name__)

class AgentCallbackHandler(BaseCallbackHandler):
    """エージェントの動作をログに記録するコールバック"""
    
    def on_agent_action(self, action, **kwargs):
        logger.info(f"ツール実行: {action.tool} - 入力: {action.tool_input}")
    
    def on_agent_finish(self, finish, **kwargs):
        logger.info(f"エージェント完了: {finish.return_values}")
    
    def on_tool_error(self, error, **kwargs):
        logger.error(f"ツールエラー: {error}")

def create_production_agent():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_retries=3,  # APIエラー時の自動リトライ
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # 本番ではFalse
        max_iterations=10,  # 無限ループ防止
        max_execution_time=60,  # 60秒でタイムアウト
        handle_parsing_errors=True,  # パースエラーを自動処理
        callbacks=[AgentCallbackHandler()],
    )
    
    return agent_executor

# 安全な実行ラッパー
async def safe_agent_run(query: str) -> dict:
    agent = create_production_agent()
    
    try:
        result = await agent.ainvoke({"input": query})
        return {"success": True, "output": result["output"]}
    except Exception as e:
        logger.error(f"エージェント実行エラー: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
```

## メモリ機能の追加

会話の文脈を維持するメモリを追加します。

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_react_agent

# 直近5ターンの会話を記憶
memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history",
    return_messages=True
)

# メモリ対応のプロンプト
prompt = hub.pull("hwchase17/react-chat")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# マルチターン会話
agent_executor.invoke({"input": "私の名前はTaroです"})
agent_executor.invoke({"input": "私の名前を覚えていますか？"})
# → "はい、Taroさんですね"
```

## 実践例：コードレビューエージェント

これらの要素を組み合わせて、実用的なコードレビューエージェントを作ります。

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain import hub
import ast
import json

@tool
def analyze_python_syntax(code: str) -> str:
    """Pythonコードの構文解析を行い、問題を特定します"""
    try:
        ast.parse(code)
        return "構文エラーなし"
    except SyntaxError as e:
        return f"構文エラー: 行{e.lineno} - {e.msg}"

@tool  
def check_security_issues(code: str) -> str:
    """セキュリティ上の問題を検出します"""
    issues = []
    
    dangerous_patterns = {
        "eval(": "evalの使用はコードインジェクションのリスクがあります",
        "exec(": "execの使用はコードインジェクションのリスクがあります",
        "os.system(": "os.systemよりsubprocessを使用してください",
        "pickle.loads(": "信頼できないデータのpickle.loadsは危険です",
        "sql = f\"": "f文字列でのSQL構築はSQLインジェクションのリスクがあります",
    }
    
    for pattern, message in dangerous_patterns.items():
        if pattern in code:
            issues.append(f"⚠️ {message}")
    
    return "\n".join(issues) if issues else "明らかなセキュリティ問題は検出されませんでした"

# エージェントシステムプロンプト
SYSTEM_PROMPT = """あなたは熟練したPythonコードレビュアーです。
コードの問題を特定し、建設的なフィードバックを提供してください。

レビューの観点:
1. セキュリティ問題（最優先）
2. パフォーマンス最適化
3. コード品質（可読性・保守性）
4. Pythonのベストプラクティス遵守

フィードバックは具体的で、修正例を含めてください。
"""

tools = [analyze_python_syntax, check_security_issues]
llm = ChatOpenAI(model="gpt-4o", temperature=0, system=SYSTEM_PROMPT)
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# レビュー実行
code_to_review = """
import pickle
import os

def process_data(user_input):
    query = f"SELECT * FROM users WHERE id = {user_input}"
    result = eval(user_input)
    data = pickle.loads(result)
    return data
"""

review = agent_executor.invoke({
    "input": f"以下のPythonコードをレビューしてください:\n\n{code_to_review}"
})
print(review["output"])
```

## まとめ

LangChainを使ったAIエージェントの基礎をカバーしました：

- **ReActパターン**による思考→行動→観察のループ
- **カスタムツール**の作成方法
- **エラーハンドリング**と本番対応
- **メモリ**による文脈維持
- 実践的な**コードレビューエージェント**の例

次のステップとして、**マルチエージェントシステム**（複数のエージェントが協調して動作する仕組み）に挑戦してみてください。次回の記事でその実装方法を解説します。
