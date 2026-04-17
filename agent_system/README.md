# AI Native Blog Publisher Agent

このシステムは、ローカルの Ollama (Gemma, etc.) を使用して、ブログ記事のトピック選定、記事生成、および GitHub への自動投稿を行う AI エージェントです。

## 機能

- **トピック選定**: 既存の記事タイトルと最新トレンドを比較し、新しい記事テーマを提案します。
- **コンテンツ生成**: `AGENTS.md` のガイドラインに従い、SEO 最適化された技術記事を生成します。
- **自動パブリッシュ**: 生成された記事をリポジトリへコミットし、GitHub へ自動的に Push します。

## 必要条件

- Go 1.24 以上
- [Ollama](https://ollama.ai/) が動作しており、設定されたモデル（デフォルト: `gemma4:26b`）が利用可能であること。
- `git` コマンドがパスに通っており、GitHub 等への Push 権限が設定されていること。

## 使い方

### ビルド

```bash
go build -o publisher ./cmd/publisher
```

### 実行

**ドライラン（Content の生成確認のみ、書き込みや Push は行わない）:**

```bash
./publisher --dry-run
```

**本番実行:**

```bash
./publisher
```

## 設定

`config.yaml` を `agent_system/` ディレクトリに配置することで設定を上書きできます。

```yaml
ollama_endpoint: "http://localhost:11434/api/generate"
model_name: "gemma4:26b"
posts_dir: "../_posts"
agents_md_path: "../AGENTS.md"
git_remote: "origin"
```

## ディレクトリ構造

- `cmd/publisher`: メインのエントリポイント
- `internal/selector`: トピック選定ロジック
- `internal/generator`: Ollama を用いた記事生成
- `internal/publisher`: Git 操作（Add, Commit, Push）
- `internal/repository`: 既存記事の解析
- `internal/config`: 設定管理
