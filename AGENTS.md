# AGENTS.md — AIエージェントによる記事執筆ガイド

このファイルはAIエージェント（Copilot、Claude、GPT-4等）がこのブログの記事を自律的に執筆するための指示書です。

## ブログの目的

**AI Native Engineer** ブログは、AIネイティブなソフトウェアエンジニアになるために必要な情報を体系的にまとめることを目的としています。

対象読者: AIを活用してソフトウェア開発の生産性・品質を向上させたいエンジニア

---

## リポジトリ構造

```
/
├── _posts/          # ブログ記事（YYYY-MM-DD-title.md 形式）
├── _pages/          # 静的ページ（about, archive, categories）
├── _layouts/        # HTMLレイアウト
├── _includes/       # 再利用可能なHTMLパーツ
├── _sass/           # スタイルシート（SCSS）
├── assets/          # 静的ファイル（CSS, JS, 画像）
├── _config.yml      # Jekyll設定
├── Gemfile          # Ruby依存関係
└── AGENTS.md        # このファイル
```

---

## 記事執筆ワークフロー

AIエージェントが記事を作成する際は、以下のステップに従ってください。

### ステップ1: 最新情報の収集

```bash
# 実行可能なツール・コマンドの例

# 1. 関連する最新論文・ニュースを収集
#    - arXiv: https://arxiv.org/search/?searchtype=all&query=LLM+agents
#    - Hacker News: https://hn.algolia.com/?q=LLM
#    - GitHub Trending: https://github.com/trending

# 2. 既存記事との重複確認
find _posts/ -name "*.md" | xargs grep -l "キーワード"

# 3. 関連するOSSや技術の最新バージョン確認
# pip show langchain / npm show openai など
```

### ステップ2: 記事の計画

以下を決定してから執筆を開始してください：

- **タイトル**: SEOキーワードを含む、具体的で魅力的なタイトル
- **カテゴリ**: `prompt-engineering` | `ai-agents` | `llm` | `architecture` | `ethics` | `tools`
- **読者レベル**: 入門 | 中級 | 上級
- **記事の種類**: チュートリアル | 解説 | 比較 | ケーススタディ | ニュース解説
- **想定読了時間**: 5〜20分（文字数: 2000〜8000字）

### ステップ3: 記事ファイルの作成

```bash
# ファイル名: YYYY-MM-DD-kebab-case-title.md
# 例:
touch _posts/2025-03-15-openai-agents-sdk-guide.md
```

---

## Front Matter テンプレート

```yaml
---
layout: post
title: "記事タイトル（日本語・SEOキーワード含む）"
description: "記事の要約（150〜160文字程度）。検索結果のスニペットとして表示されます。"
date: YYYY-MM-DD HH:MM:SS +0900
categories: [カテゴリ名]  # 上記カテゴリリストから1〜2個
tags: [タグ1, タグ2, タグ3]  # 関連キーワード
author: "AI Native Engineer"
reading_time: X  # 分（200文字/分で計算）
image: /assets/images/posts/記事スラッグ.png  # OGP画像（任意）
---
```

---

## 記事執筆ルール

### 必須要件

1. **日本語で執筆する**（コードコメントも日本語推奨）
2. **実際に動作するコード例を含める**（ハルシネーションに注意）
3. **出典・参考文献を記載する**（特に技術的な主張）
4. **図表や比較表を積極的に使う**（Markdownテーブル形式）

### 記事構成パターン

#### チュートリアル記事
```
## はじめに（なぜこれが重要か・読者が得るもの）
## 前提知識・環境
## ステップ1: ○○
## ステップ2: ○○
## ステップ3: ○○
## よくある問題と解決策
## まとめ（学んだこと・次のステップ）
```

#### 解説記事
```
## ○○とは何か（定義・概要）
## なぜ重要か（背景・問題意識）
## 仕組みの解説（コアコンセプト）
## 実践的な活用法
## 注意点・制限事項
## まとめ
```

#### 比較記事
```
## 比較の背景（なぜ比較するか）
## 各ツール/手法の概要
## 比較表（機能・コスト・パフォーマンス）
## ユースケース別おすすめ
## まとめ
```

### コードブロックのルール

```markdown
# 言語を必ず指定する
\`\`\`python
# コードは動作するものを使用
\`\`\`

# インストールコマンドも含める
\`\`\`bash
pip install パッケージ名==バージョン
\`\`\`
```

### SEO最適化

- **タイトル**: 主要キーワードを先頭に（60文字以内）
- **description**: 検索意図を満たす要約（155文字以内）
- **見出し構造**: H2を4〜6個、H3を適宜使用
- **内部リンク**: 関連する既存記事へのリンクを含める
- **画像alt**: 画像には必ずalt属性を記述

---

## カテゴリ別 執筆ガイドライン

### `prompt-engineering`
- Chain-of-Thought、Few-shot、Function Calling等の技術
- プロンプトテンプレートのサンプルを含める
- 「良い例」と「悪い例」を対比させる
- 実際のAPIレスポンス例を含める

### `ai-agents`
- エージェントの設計図・アーキテクチャを示す
- ツール定義のコード例を含める
- エラーハンドリングのベストプラクティスを記載
- 実際の動作ログやトレースを示す

### `llm`
- RAG、ファインチューニング、評価方法等
- コスト比較表を含める（可能な場合）
- ベンチマーク結果や比較データを示す
- オープンソースとクローズドモデルの両方を扱う

### `architecture`
- システム構成図を含める（Mermaidダイアグラム推奨）
- スケーラビリティの考慮事項を記載
- 実際の本番事例・パターン名を引用する

### `ethics`
- 具体的な事例を基に議論する
- バイアス、プライバシー、安全性の各側面を扱う
- 対処法・緩和策を提案する

### `tools`
- インストール方法から使い方まで一通りカバー
- 競合ツールとの比較を含める
- バージョン情報を記載（陳腐化しやすい）

---

## Mermaidダイアグラムの使い方

```markdown
\`\`\`mermaid
graph TD
    A[ユーザー入力] --> B[入力バリデーション]
    B --> C{有効?}
    C -->|Yes| D[LLM呼び出し]
    C -->|No| E[エラー返却]
    D --> F[出力バリデーション]
    F --> G[ユーザーへの応答]
\`\`\`
```

---

## 品質チェックリスト

記事を公開前に以下を確認してください：

### コンテンツ
- [ ] タイトルとdescriptionがSEO最適化されている
- [ ] コード例が正しく動作する（手動またはCI/CDで検証）
- [ ] 技術的な主張に根拠がある（出典リンクまたは実験結果）
- [ ] 日本語の表現が自然（機械的な直訳になっていない）
- [ ] 内部リンクが2〜3個含まれている

### 技術
- [ ] Front MatterのYAMLが正しい
- [ ] カテゴリが既存のリストから選択されている
- [ ] ファイル名がYYYY-MM-DD-slug.md 形式
- [ ] コードブロックに言語指定がある

### 倫理・正確性
- [ ] ハルシネーション（事実誤認）がないことを確認
- [ ] 記事作成にAIが使用されていることを開示（任意）
- [ ] 特定製品・企業の過度な宣伝になっていない

---

## 記事作成コマンド例

```bash
# 新しい記事を作成するワンライナー
DATE=$(date +%Y-%m-%d)
SLUG="your-article-slug"
cat > _posts/${DATE}-${SLUG}.md << 'EOF'
---
layout: post
title: ""
description: ""
date: ${DATE} 10:00:00 +0900
categories: []
tags: []
author: "AI Native Engineer"
reading_time: 10
---

## はじめに

EOF

# Jekyll開発サーバーで確認
bundle exec jekyll serve --livereload

# ビルドのみ
bundle exec jekyll build
```

---

## 自律的な情報収集のヒント

AIエージェントが定期的に情報収集する際の推奨ソース：

### 技術ニュース
- **Hacker News**: https://news.ycombinator.com/
- **The Batch (DeepLearning.AI)**: https://www.deeplearning.ai/the-batch/
- **Import AI**: https://jack-clark.net/
- **LLM Papers**: https://github.com/Hannibal046/Awesome-LLM

### 論文
- **arXiv cs.AI**: https://arxiv.org/list/cs.AI/recent
- **arXiv cs.CL**: https://arxiv.org/list/cs.CL/recent
- **Papers With Code**: https://paperswithcode.com/

### ツール・ライブラリ
- **LangChain Changelog**: https://github.com/langchain-ai/langchain/releases
- **LlamaIndex**: https://github.com/run-llama/llama_index/releases
- **OpenAI Cookbook**: https://github.com/openai/openai-cookbook
- **Anthropic Cookbook**: https://github.com/anthropics/anthropic-cookbook

### コミュニティ
- **r/LocalLLaMA**: https://www.reddit.com/r/LocalLLaMA/
- **Discord: Eleuther AI / LangChain**

---

## 注意事項

1. **著作権に注意**: 他のブログや論文からの無断コピーは禁止。要約・引用は出典を明記。
2. **コードのライセンス**: サンプルコードのライセンスを確認し、適切に引用する。
3. **個人情報**: 実際の個人情報やAPIキーをコード例に含めない。
4. **最新性の確認**: ライブラリのバージョンや機能は急速に変わる。執筆時点のバージョンを明記する。
5. **ハルシネーション対策**: 技術的な事実（パフォーマンス数値、API仕様等）は必ず公式ドキュメントで確認する。

---

*このファイルはAIエージェントが参照するための指示書です。ブログの方針変更や新しいカテゴリの追加があった場合は、このファイルを更新してください。*
