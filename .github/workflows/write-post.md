---
on:
  schedule: daily
permissions:
      contents: read
      issues: read
      pull-requests: read
engine: copilot
network:
  allowed:
    - defaults
    - python
    - node
    - go
    - java
tools:
  github:
    toolsets: [default]
  web-fetch:
safe-outputs:
  create-pull-request:
---

# write-post

あなたはプロのライターです。AIネイティブなエンジニアになるためのTips記事を、今日の日付で執筆してください。

過去の記事のテーマも参考にしつつ、生成AIの活用方法や上級者向けのテクニック、AIのしくみやAIの最新情報など、様々なトピックを扱ってください。
