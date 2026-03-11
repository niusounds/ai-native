# ai-native

AIネイティブエンジニア向けブログ。

## デプロイ

### Cloudflare Pages

Cloudflare Pages ダッシュボードのビルド設定を以下のように設定してください：

| 項目 | 設定値 |
|------|--------|
| Framework preset | `Jekyll` |
| Build command | `bundle exec jekyll build --config _config.yml,_config_cloudflare.yml` |
| Build output directory | `_site` |

`_config.yml` には GitHub Pages 用の `baseurl: "/ai-native"` が設定されています。
Cloudflare Pages はドメインのルートにデプロイされるため、`_config_cloudflare.yml` で
`baseurl: ""` を上書きする必要があります。この上書きをしないと、CSS/JS ファイルのパスが
`/ai-native/assets/...` となり、Cloudflare Pages 上では 404 (HTML) が返されるため
MIME タイプエラーが発生します。

### GitHub Pages

`_config.yml` のデフォルト設定（`baseurl: "/ai-native"`）でそのままビルドできます。