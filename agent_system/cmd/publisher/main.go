package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

func main() {
	var dryRun bool

	var rootCmd = &cobra.Command{
		Use:   "publisher",
		Short: "AI Native Blog Post Publisher",
		Long:  `An automated agent system that selects topics, generates AI-related blog posts using Ollami, and publishes them to GitHub.`,
		Run: func(cmd *cobra.Command, args []string) {
			if dryRun {
				fmt.Println("Running in DRY-RUN mode. No changes will be made to the filesystem or git repository.")
			}
			fmt.Println("Starting the publishing process...")
			// TODO: Orchestrate the components: Selector -> Generator -> Publisher
		},
	}

	rootCmd.PersistentFlags().BoolVarP(&dryRun, "dry-run", "d", false, "Run without making any actual changes")

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
