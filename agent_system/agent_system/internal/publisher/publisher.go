package publisher

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/object"
)

type GitPublisher struct {
	repoPath string
	remote   string
}

func NewGitPublisher(repoPath, remote string) *GitPublisher {
	return &Gitpsublisher{
		repoPath: repoPath,
		remote:   remote,
	}
}

func (p *GitPublisher) Publish(filePath string, message string) error {
	repo, err := git.PlainOpen(p.repoPath)
	if err != nil {
		return fmt.Errorf("failed to open repo: %w", err)
	}

	worktree, err := repo.Worktree()
	if err != nil {
		return fmt.Errorf("failed to get worktree: %w", err)
	}

	_, err = worktree.Add(filepath.Base(filePath))
	if err != nil {
		return fmt.Errorf("failed to add file: %w", err)
	}

	_, err = worktree.Commit(message, &git.CommitOptions{
		Author: &object.Signature{
			Name:  "AI Native Agent",
			Email: "agent@ai-native.com",
			When:  time.Now(),
		},
	})
	if err != nil {
		return fmt.Errorf("failed to commit: %w", err)
	}

	// In a real implementation, we would push to the remote.
	// For this task, we'll just log it.
	fmt.Printf("[Git] Successfully committed: %s\n", message)
	fmt.Printf("[Git] Ready to push to %s\n", p.remote)

	return nil
}
