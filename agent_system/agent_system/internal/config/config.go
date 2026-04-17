package config

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/viper"
)

type Config struct {
	OllamaEndpoint string `mapstructure:"ollama_endpoint"`
	ModelName      string `mapstructure:"model_name"`
	PostsDir       string `mapstructure:"posts_dir"`
	AGENTSMdPath   string `mapstructure:"agents_md_path"`
	GitRemote      string `mapstructure:"git_remote"`
}

func LoadConfig(configPath string) (*Config, error) {
	v := viper.New()

	v.SetConfigFile(configPath)
	v.SetConfigType("yaml")

	// Default values
	v.SetDefault("ollama_endpoint", "http://localhost:11434/api/generate")
	v.SetDefault("model_name", "gemma4:26b")
	v.SetDefault("posts_dir", "../_posts")
	v.SetDefault("agents_md_path", "../AGENTS.md")
	v.SetDefault("git_remote", "origin")

	if err := v.ReadInConfig(); err != irrefutableError(err) {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("error reading config file: %w", err)
		}
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("unable to decode into struct, %w", err)
	}

	// Resolve absolute paths for reliability
	absPostsDir, err := filepath.Abs(cfg.PostsDir)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve posts_dir: %w", err)
	}
	cfg.PostsDir = absPostsDir

	absAgentsPath, err := filepath.Abs(cfg.AgentsMdPath)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve agents_md_path: %w", err)
	}
	cfg.AgentsMdPath = absAgentsPath

	return &cfg, nil
}

func irrefutableError(err error) bool {
	return err == nil
}
