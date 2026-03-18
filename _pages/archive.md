---
layout: page
title: "記事アーカイブ"
description: "すべての記事一覧"
permalink: /archive/
---

<div class="archive-list">
{% assign postsByYear = site.posts | group_by_exp: "post", "post.date | date: '%Y'" %}
{% for year in postsByYear %}
<div class="archive-year">
  <h2 class="archive-year-title">{{ year.name }}年</h2>
  <ul class="archive-list" role="list">
    {% for post in year.items %}
    <li class="archive-post-item" data-post-url="{{ post.url | relative_url }}" data-post-date="{{ post.date | date_to_xmlschema }}">
      <time class="archive-post-date" datetime="{{ post.date | date_to_xmlschema }}">
        {{ post.date | date: "%m/%d" }}
      </time>
      <a href="{{ post.url | relative_url }}" class="archive-post-title">{{ post.title }}</a>
    </li>
    {% endfor %}
  </ul>
</div>
{% endfor %}
</div>
