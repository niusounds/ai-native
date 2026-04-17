package selector

import (
	"context"
	"fmt"
	"os"
	"strings"

	"agent_system/internal/config"
	"agent_system/internal/repository"
	"github.com/gocolly/colly/v2"
)

type TopicSelector struct {
	cfg          *config.Config
	repo         *repository.PostRepository
	ollamaClient *OllamaClient // Interface or real client
}

type OllamaClient interface {
	Generate(ctx context.Context, prompt string) (string, error)
}

func NewTopicSelector(cfg *config.Config, repo *repository.PostRepository, client OllamaClient) *TopicSelector {
	return &TopicSelector{
		cfg:          cfg,
		repo:         repo,
		ollamaClient: client,
	}
	}
}

func (s *TopicSelector) SelectNextTopic(ctx context.Context) (string, error) {
	// 1. Get existing titles to avoid duplication
	existingTitles, err := s.repo.GetAllPostTitles()
	if err != nil {
		return "", fmt.Errorf("failed to get existing titles: %w", err)
	}

	// 2. Scrape news (Simplified: using a placeholder for actual scraping logic)
	// In production, use colly to scrape HN or arXiv
	news := "Recent trends in AI: Agentic workflows, small language models, and multimodal reasoning."

	// 3. Use LLM to brainstorm a new topic
	prompt := fmt.Sprintf(`
You are a trend researcher for the "AI Native Engineer" blog.
Existing blog post titles:
%v

Current news/trends:
%s

Based on the existing posts and the new trends, suggest ONE highly specific, technical, and engaging blog post topic in Japanese.
The topic must NOT overlap with the existing titles.
Return only the topic title.
`, existingTitles, news)

	topic, err := s.ollamaClient.Generate(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("failed to generate topic: %w", err)
	}

	return strings.TrimSpace(topic), nil
}
