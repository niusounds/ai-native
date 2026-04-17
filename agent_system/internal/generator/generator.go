package generator

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"ai-native-publisher/internal/config"
)

type OllamaClient struct {
	endpoint string
	model    string
	client   *http.Client
}

func NewOllamaClient(cfg *config.Config) *OllamaClient {
	return &OllamaClient{
		endpoint: cfg.OllamaEndpoint,
		model:    cfg.ModelName,
		client:   &http.Client{},
	}
}

type GenerateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type GenerateResponse struct {
	Response string `json:"response"`
}

func (c *OllamaClient) Generate(ctx context.Context, prompt string) (string, error) {
	reqBody, err := json.Marshal(GenerateRequest{
		Model:  c.model,
		Prompt: prompt,
		Stream: false,
	})
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.endpoint, bytes.NewBuffer(reqBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama returned non-OK status: %d", resp.StatusCode)
	}

	var genResp GenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	return genResp.Response, nil
}

type ContentGenerator struct {
	client *OllamaClient
	agentsMd string
}

func NewContentGenerator(client *OllamaClient, agentsMd string) *ContentGenerator {
	return &ContentGenerator{
		client: client,
		agentsMd: agentsMd,
	}
}

func (g *ContentGenerator) GeneratePost(ctx context.Context, topic string) (string, error) {
	prompt := fmt.Sprintf(`
あなたは「AI Native Engineer」ブログの専属執筆エージェントです。
以下の「AGENTS.md」に記載された執筆ガイドライン、構成案、SEOルールを完全に遵守して、最高品質の記事を作成してください。

# コンテキスト: AGENTS.md (執筆ルール)
%s

# 今日の執筆タスク
- トピック: %s
- 読者ターゲット: エンジニア（中級〜上級）
- 言語: 日本語

# 執筆上の重要ルール
1. **Front Matter**: Jekyllが解釈可能な形式で必ず含めてください。
2. **コード例**: 必ず動作する詳細なコード例（Python, JavaScript, Go, Rust等）を含めてください。
3. **図解**: Mermaid形式（コードブロックの言語に mermaid を指定）でシステムのアーキテクチャやフローを示してください。
4. **内部リンク**: 既存の記事への言及を含め、回遊性を高めてください。
5. **文体**: 専門的かつ誠実なエンジニアらしいトーンで記述してください。
6. **ファイル名**: 記事の最後に「FILENAME:YYYY-MM-DD-slug.md」の形式で、推奨されるファイル名を1行だけ記述してください。

執筆を開始してください：
`, g.agentsMd, topic)

	return g.client.Generate(ctx, prompt)
}
