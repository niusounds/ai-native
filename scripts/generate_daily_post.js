const fs = require('fs');
const path = require('path');

/**
 * AI Native Blog - Daily Post Generator
 * Using Ollama and Gemma
 */

const OLLAMA_URL = 'http://localhost:11434/api/generate';
const MODEL = 'gemma4:26b'; // ユーザー指定のモデル名

async function generatePost() {
    const agentsMdPath = path.join(__dirname, '../AGENTS.md');
    const postsDir = path.join(__dirname, '../_posts');

    if (!fs.existsSync(agentsMdPath)) {
        console.error('Error: AGENTS.md が見つかりません。プロジェクトのルートで実行するか、パスを確認してください。');
        process.exit(1);
    }

    const agentsMd = fs.readFileSync(agentsMdPath, 'utf8');
    
    // 引数からトピックを取得（デフォルト値あり）
    const topic = process.argv[2] || "AIエージェントの自律化における最新トレンドと課題";

    const date = new Date();
    const dateStr = date.toISOString().split('T')[0];

    // プロンプトの構築（AGENTS.mdのルールを注入）
    const prompt = `
あなたは「AI Native Engineer」ブログの専属執筆エージェントです。
以下の「AGENTS.md」に記載された執筆ガイドライン、構成案、SEOルールを完全に遵守して、最高品質の記事を作成してください。

# コンテキスト: AGENTS.md (執筆ルール)
${agentsMd}

# 今日の執筆タスク
- トピック: ${topic}
- 公開日: ${dateStr}
- 読者ターゲット: エンジニア（中級〜上級）
- 言語: 日本語

# 執筆上の重要ルール
1. **Front Matter**: Jekyllが解釈可能な形式で必ず含めてください。
2. **コード例**: 必ず動作する詳細なコード例（Python, JavaScript, Go, Rust等）を含めてください。
3. **図解**: Mermaid形式（\`\`\`mermaid\`\`\`）でシステムのアーキテクチャやフローを示してください。
4. **内部リンク**: 既存の記事（reasoning-models, llm-evals, memory-system, finetuning-lora, observability-guide）への言及を含め、回遊性を高めてください。
5. **文体**: 専門的かつ誠実なエンジニアらしいトーンで記述してください。
6. **ファイル名**: 記事の最後に「FILENAME: YYYY-MM-DD-slug.md」の形式で、推奨されるファイル名を1行だけ記述してください。

執筆を開始してください：
`;

    console.log(`[Status] 記事生成を開始します...`);
    console.log(`[Config] Topic: "${topic}"`);
    console.log(`[Config] Model: ${MODEL}`);

    try {
        const response = await fetch(OLLAMA_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: MODEL,
                prompt: prompt,
                stream: false,
                options: {
                    num_ctx: 16384 // コンテキスト長を広めに確保
                }
            })
        });

        if (!response.ok) {
            throw new Error(`Ollama API エラー: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        let content = data.response;

        if (!content) {
            throw new Error('モデルからの応答が空です。');
        }

        // ファイル名の抽出
        const filenameMatch = content.match(/FILENAME:\s*(.+)/);
        let filename = filenameMatch ? filenameMatch[1].trim() : `${dateStr}-generated-post.md`;
        
        // ファイル名指定行を除去
        content = content.replace(/FILENAME:\s*.+/, '').trim();
        
        // Markdownブロックで囲まれている場合の除去
        content = content.replace(/^```markdown\n/, '').replace(/\n```$/, '');

        // _posts ディレクトリの確認と保存
        if (!fs.existsSync(postsDir)) {
            fs.mkdirSync(postsDir, { recursive: true });
        }

        const filePath = path.join(postsDir, filename);
        fs.writeFileSync(filePath, content, 'utf8');

        console.log(`\n[Success] 記事の生成に成功しました！`);
        console.log(`[Path] ${filePath}`);
        console.log(`[Filename] ${filename}`);
        console.log(`\nJekyllサーバーを起動して確認してください: bundle exec jekyll serve`);

    } catch (error) {
        console.error(`\n[Error] 記事生成中にエラーが発生しました:`);
        console.error(error.message);
        if (error.message.includes('fetch')) {
            console.log('\nヒント: Ollamaが起動しているか、モデル "' + MODEL + '" がプルされているか確認してください。');
        }
    }
}

generatePost();
