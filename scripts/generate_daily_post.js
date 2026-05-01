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
    /**
     * Returns an array of titles from existing posts in _posts directory.
     */
    function getPastPostTitles() {
        const titles = [];
        if (fs.existsSync(postsDir)) {
            const files = fs.readdirSync(postsDir);
            files.forEach(file => {
                if (file.endsWith('.md')) {
                    const fullPath = path.join(postsDir, file);
                    const content = fs.readFileSync(fullPath, 'utf8');
                    const match = content.match(/^title:\s*["']?(.*?)["']?$/m);
                    if (match) {
                        titles.push(match[1].trim());
                    }
                }
            });
        }
        return titles;
    }

    // トピックはLLMに自動生成させる（引数で指定可能）
    let topic = process.argv[2];

    const date = new Date();
    const dateStr = date.toISOString().split('T')[0];

    // ファイル名の抽出前に過去記事タイトルを取得
    const pastTitles = getPastPostTitles();
    const pastList = pastTitles.length ? pastTitles.map(t => `- ${t}`).join('\n') : '（過去記事なし）';

    // プロンプトの構築（AGENTS.mdのルールを注入）
    const prompt = `
あなたは「AI Native Engineer」ブログの専属執筆エージェントです。

## トピックについて
- トピックが既に指定されている場合は、そのテーマで記事を作成してください。
- トピックが未指定の場合は、以下の基準で自分で最適なトピックを1つ選定してください:
  1. **最新性**: 2025-2026年のAI関連技術・トレンドに関連するもの
  2. **実用性**: エンジニアが実際に活用できる具体的な知見を含めるもの
  3. **独自性**: 過去の投稿タイトルと重複しないもの
  4. **深さ**: 中級〜上級エンジニアにとって価値のある技術的深みがあるもの

# 過去の記事タイトル
${pastList}

# コンテキスト: AGENTS.md (執筆ルール)
${agentsMd}

# 今日の執筆タスク
- トピック: ${topic || '自分で最適なトピックを1つ選定して決定してください。AI関連の最新技術・トレンドから、エンジニアにとって価値のある主题を選んでください'}
- 公開日: ${dateStr}
- 選択したトピック: ${topic ? '' : '（上記で自己決定）'}
- 読者ターゲット: エンジニア（中級〜上級）
- 言語: 日本語

# 執筆上の重要ルール
1. **出力形式**: 記事は **---**（フロントマター区切り記号）から直接始めてください。 planningセクション、メタ情報、要約、またはフロントマター以外の本文を記事の先頭に含めないでください。
2. **Front Matter**: Jekyllが解釈可能な形式で必ず含めてください。
3. **コード例**: 必ず動作する詳細なコード例を含めてください。変数名の誤字脱字に注意してください。
4. **図解**: Mermaid形式でシステムのアーキテクチャやフローを示してください。
5. **内部リンク**: 既存記事への言及は実際のURLパス形式で記述してください。プレースホルダ形式は禁止です。
6. **文体**: 専門的かつ誠実なエンジニアらしいトーンで記述してください。
7. **ファイル名**: 記事の最後に「FILENAME: YYYY-MM-DD-slug.md」の形式で、推奨されるファイル名を1行だけ記述してください。

# 出力の厳格なルール
- 記事の出力は「---」から始めてください
- 執筆計画、タイトル案、カテゴリ案などのメタデータは出力しないでください
- 記事の本文はフロントマター（---区切り）の外には一切記述しないでください

執筆を開始してください：
`;

    console.log(`[Status] 記事生成を開始します...`);
    console.log(`[Config] Topic: ${topic || '(LLMが自動選定)'}`);
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

        // 記事の先頭にある planning/メタセクションを除去
        // 執筆の計画、タイトル案、カテゴリ案などのプレテキスト
        content = content.replace(/^(#\s*執筆の計画[\s\S]*?)(?=\n---)/, '').trim();
        content = content.replace(/^(#\s*Planning[\s\S]*?)(?=\n---)/, '').trim();
        content = content.replace(/^(##\s*Plan[\s\S]*?)(?=\n---)/, '').trim();

        // 先頭が --- で始まらない場合は、最初の見出しから --- の間を除去
        if (!content.startsWith('---')) {
            const firstFrontMatter = content.indexOf('---');
            if (firstFrontMatter > 0) {
                content = content.substring(firstFrontMatter);
            }
        }

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
