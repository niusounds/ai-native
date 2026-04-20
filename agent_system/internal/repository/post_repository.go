package repository

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

type PostMetadata struct {
	Title    string   `yaml:"title"`
	Date     string   `yaml:"date"`
	Category []string `yaml:"categories"`
	Tags     []string `yaml:"tags"`
}

type PostRepository struct {
	postsDir string
}

func NewPostRepository(postsDir string) *PostRepository {
	return &PostRepository{postsDir: postsDir}
}

func (r *PostRepository) GetAllPostTitles() ([]string, error) {
	files, err := os.ReadDir(r.postsDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read posts directory: %w", err)
	}

	var titles []string
	for _, file := range files {
		if !file.IsDir() && strings.HasSuffix(file.Name(), ".md") {
			title, err := r.getPostTitle(filepath.Join(r.postsDir, file.Name()))
			if err == nil {
				titles = append(titles, title)
			}
		}
	}
	return titles, nil
}

func (r *PostRepository) getPostTitle(filePath string) (string, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}

	// Simple frontmatter parsing
	parts := strings.Split(string(data), "---")
	if len(parts) < 3 {
		return "", fmt.Errorf("invalid markdown format in %s", filePath)
	}

	var metadata PostMetadata
	if err := yaml.Unmarshal([]byte(parts[1]), &metadata); err != nil {
		return "", fmt.Errorf("failed to unmarshal frontmatter in %s: %w", filePath, err)
	}

	if metadata.Title == "" {
		return "", fmt.Errorf("missing title in %s", filePath)
	}

	return metadata.Title, nil
}

