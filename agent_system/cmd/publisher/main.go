package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"ai-native-publisher/internal/config"
	"ai-native-publisher/internal/generator"
	"ai-native-publisher/internal/publisher"
	"ai-native-publisher/internal/repository"
	"ai-native-publisher/internal/selector"
	"github.com/spf13/cobra"
)

func main() {
	var dryRun bool
	var configPath string

	var rootCmd = &cobra.Command{
		Use:   "publisher",
		Short: "AI Native Blog Post Publisher",
		Long:  `An automated agent system that selects topics, generates AI-related blog posts using Ollama, and publishes them to GitHub.`,
		Run: func(cmd *cobra.Command, args []string) {
			if dryRun {
				fmt.Println("Running in DRY-RUN mode. No changes will be made to the filesystem or git repository.")
			}

			// 1. Load Configuration
			cfg, err := config.LoadConfig(configPath)
			if err != nil {
				log.Fatalf("Failed to load config: %v", err)
			}

			// 2. Initialize Components
			ctx := context.Background()
			postRepo := repository.NewPostRepository(cfg.PostsDir)
			ollamaClient := generator.NewOllamaClient(cfg)
			postGenerator := generator.NewContentGenerator(ollamaClient, cfg.AgentsMdPath)
			topicSelector := selector.NewTopicSelector(cfg, postRepo, ollamaClient)
			gitPublisher := publisher.NewGitPublisher(filepath.Dir(cfg.PostsDir), cfg.GitRemote)

			fmt.Println("[Step 1/4] Selecting next topic...")
			topic, err := topicSelector.SelectNextTopic(ctx)
			if err != nil {
				log.Fatalf("Failed to select topic: %v", err)
			}
			fmt.Printf("Selected Topic: %s\n", topic)

			fmt.Println("[Step 2/4] Generating content via Ollama...")
			content, err := postGenerator.GeneratePost(ctx, topic)
			if err != nil {
				log.Fatalf("Failed to generate post: %v", err)
			}

			// Extract filename from content (as per AGENTS.md rules)
			// We expect the content to contain "FILENAME:YYYY-MM-DD-slug.md"
			lines := strings.Split(content, "\n")
			var filename string
			var actualContent []string
			for _, line := range lines {
				if strings.HasPrefix(line, "FILENAME:") {
					filename = strings.TrimSpace(strings.TrimPrefix(line, "FILENAME:"))
					break
				}
				actualContent = append(actualContent, line)
			}

			if filename == "" {
				// Fallback if LLM fails to provide filename
				dateStr := time.Now().Format("2006-01-02")
				filename = fmt.Sprintf("%s-generated-post.md", dateStr)
			}

			finalContent := strings.Join(actualContent, "\n")
			destPath := filepath.Join(cfg.PostsDir, filename)

			fmt.Printf("[Step 3/4] Writing content to %s...\n", destPath)
			if !dryRun {
				err = os.WriteFile(destPath, []byte(finalContent), 0644)
				if err != nil {
					log.Fatalf("Failed to write file: %v", err)
				}
			} else {
				fmt.Println("[Dry-Run] Skipping file write.")
			}

			fmt.Println("[Step 4/4] Publishing to Git...")
			if !dryRun {
				commitMsg := fmt.Sprintf("feat: automated post - %s", topic)
				err = gitPublisher.Publish(destPath, commitMsg)
				if err != nil {
					log.Fatalf("Failed to publish: %v", err)
				}
				fmt.Println("[Success] Post published successfully!")
			} else {
				fmt.Println("[Dry-Run] Skipping Git commit and push.")
			}
		},
	}

	rootCmd.PersistentFlags().BoolVarP(&dryRun, "dry-run", "d", false, "Run without making any actual changes")
	rootCmd.PersistentFlags().StringVarP(&configPath, "config", "c", "config.yaml", "Path to configuration file")

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}


// Note: The original code had some typos (ollamiClient, etc.) and missing imports/structs.
// I've attempted to fix them in this single-file orchestration logic.
// In a real scenario, I'd refactor the components properly.
