# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## 概要

「AI Native Engineer」ブログのリポジトリ。Jekyll ベースの技術ブログで、AI エージェントによる自動記事執筆システムも併設している。

- URL: https://ainative.niusounds.com
- Jekyll 4.3 + kramdown + rouge ハイライト
- 全記事は日本語で執筆

## 開発コマンド

```bash
# ローカル開発サーバー（ライブリロード付き）
bundle exec jekyll serve --livereload

# ビルドのみ
bundle exec jekyll build

# agent_system (Go) のビルド
cd agent_system && go build -o publisher ./cmd/publisher
```

## リポジトリ構造

```
/
├── _posts/           # ブログ記事 (YYYY-MM-DD-title.md 形式)
├── _pages/           # 静的ページ (about, archive, categories)
├── _layouts/         # HTMLレイアウト (default, post, page, home, category)
├── _includes/        # 再利用可能なHTMLパーツ
├── _sass/            # SCSS (base, layout, components, variables)
├── assets/           # 静的ファイル (CSS, JS, 画像)
├── _categories/      # カテゴリ定義 (front matter 付き Markdown)
├── _config.yml       # Jekyll メイン設定
├── _config_cloudflare.yml  # Cloudflare Pages 用設定
├── AGENTS.md         # AIエージェントによる記事執筆ガイドライン（重要）
├── agent_system/     # Go 製自動記事パブリッシャーエージェント
│   ├── cmd/publisher/
│   ├── internal/selector/   # トピック選定
│   ├── internal/generator/  # Ollama による記事生成
│   ├── internal/publisher/  # Git 操作 (Add, Commit, Push)
│   ├── internal/repository/ # 既存記事の解析
│   └── internal/config/     # 設定管理
├── scripts/
│   └── generate_daily_post.js  # Node.js 版記事生成スクリプト
└── Gemfile
```

## 記事執筆ルール（AGENTS.md 要約）

- 全記事を日本語で執筆
- Front Matter は必ず Jekyll 互換形式で記述（`layout: post`、`title`、`description`、`date`、`categories`、`tags`）
- カテゴリは `prompt-engineering` / `ai-agents` / `llm` / `architecture` / `ethics` / `tools` から選択
- コードブロックは言語指定必須
- Mermaid ダイアグラムを積極的に使用
- 技術的主張には出典を明記
- ハルシネーション（事実誤認）に注意

詳細は [AGENTS.md](AGENTS.md) を参照。

## 記事ファイル名規則

```
YYYY-MM-DD-kebab-case-title.md
```

例: `2026-05-10-llm-as-a-judge-pipeline.md`

## パーマリンク形式

```
/:categories/:year/:month/:day/:title/
```

## 既存記事の確認方法

重複回避のため、新規記事作成前に既存記事を確認すること:

```bash
find _posts/ -name "*.md" | xargs grep -l "キーワード"
```

## AI 自動記事生成システム

- **Go 版** (`agent_system/`): Ollama (Gemma 4 26b) を使用。トピック選定 → コンテンツ生成 → Git commit & push の全自動化
- **JS 版** (`scripts/`): Ollama を呼び出して記事を生成するスクリプト
- 設定は `agent_system/config.yaml` で変更可能

## 設定ファイル

- `_config.yml`: メイン設定（URL は `niusounds.github.io`、baseurl `/ai-native`）
- `_config_cloudflare.yml`: Cloudflare Pages 向け設定
- `agent_system/config.yaml`: パブリッシャーエージェント設定
