// AI Native Engineer Blog - Main JavaScript

(function () {
  'use strict';

  // ============================================
  // Mobile Navigation Toggle
  // ============================================
  const navToggle = document.querySelector('.nav-toggle');
  const navMenu = document.querySelector('.nav-menu');

  if (navToggle && navMenu) {
    navToggle.addEventListener('click', function () {
      const isOpen = navMenu.classList.toggle('is-open');
      navToggle.setAttribute('aria-expanded', isOpen.toString());
      navToggle.setAttribute('aria-label', isOpen ? 'メニューを閉じる' : 'メニューを開く');
    });

    // Close menu on outside click
    document.addEventListener('click', function (e) {
      if (!navToggle.contains(e.target) && !navMenu.contains(e.target)) {
        navMenu.classList.remove('is-open');
        navToggle.setAttribute('aria-expanded', 'false');
        navToggle.setAttribute('aria-label', 'メニューを開く');
      }
    });

    // Close menu on Escape key
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && navMenu.classList.contains('is-open')) {
        navMenu.classList.remove('is-open');
        navToggle.setAttribute('aria-expanded', 'false');
        navToggle.focus();
      }
    });
  }

  // ============================================
  // Table of Contents Generator
  // ============================================
  const tocNav = document.getElementById('toc-nav');
  const postContent = document.querySelector('.post-content');

  if (tocNav && postContent) {
    const headings = postContent.querySelectorAll('h2, h3');

    if (headings.length > 0) {
      const tocList = document.createDocumentFragment();

      headings.forEach(function (heading, index) {
        // Ensure heading has an ID
        if (!heading.id) {
          const id = heading.textContent
            .trim()
            .toLowerCase()
            .replace(/[^\w\s-]/g, '')
            .replace(/\s+/g, '-')
            .replace(/-+/g, '-')
            .substring(0, 60);
          heading.id = id + '-' + index;
        }

        const link = document.createElement('a');
        link.href = '#' + heading.id;
        link.textContent = heading.textContent;
        link.className = heading.tagName === 'H3' ? 'toc-h3' : '';

        link.addEventListener('click', function (e) {
          e.preventDefault();
          const target = document.getElementById(heading.id);
          if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            history.replaceState(null, '', '#' + heading.id);
          }
        });

        tocList.appendChild(link);
      });

      tocNav.appendChild(tocList);

      // ---- Active TOC link on scroll ----
      const observer = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (entry) {
            const id = entry.target.id;
            const link = tocNav.querySelector('a[href="#' + id + '"]');
            if (link) {
              if (entry.isIntersecting) {
                tocNav.querySelectorAll('a').forEach(function (a) {
                  a.classList.remove('active');
                });
                link.classList.add('active');
              }
            }
          });
        },
        {
          rootMargin: '-64px 0px -80% 0px',
          threshold: 0,
        }
      );

      headings.forEach(function (h) {
        observer.observe(h);
      });
    }
  }

  // ============================================
  // Lazy loading images (native + fallback)
  // ============================================
  if ('loading' in HTMLImageElement.prototype) {
    // Native lazy loading supported
    document.querySelectorAll('img[loading="lazy"]').forEach(function (img) {
      if (img.dataset.src) {
        img.src = img.dataset.src;
      }
    });
  } else {
    // Fallback: IntersectionObserver
    const lazyImages = document.querySelectorAll('img[data-src]');
    if (lazyImages.length > 0) {
      const imageObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            const img = entry.target;
            img.src = img.dataset.src;
            img.removeAttribute('data-src');
            imageObserver.unobserve(img);
          }
        });
      });
      lazyImages.forEach(function (img) {
        imageObserver.observe(img);
      });
    }
  }

  // ============================================
  // Smooth scroll for anchor links
  // ============================================
  document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
    anchor.addEventListener('click', function (e) {
      const href = anchor.getAttribute('href');
      if (href === '#') return;
      const target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.replaceState(null, '', href);
      }
    });
  });

  // ============================================
  // Copy code button
  // ============================================
  document.querySelectorAll('pre').forEach(function (pre) {
    if (!navigator.clipboard) return;

    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.setAttribute('aria-label', 'コードをコピー');
    btn.innerHTML =
      '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">' +
      '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>' +
      '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>' +
      '</svg>';

    pre.style.position = 'relative';
    pre.appendChild(btn);

    btn.addEventListener('click', function () {
      const code = pre.querySelector('code');
      const text = code ? code.textContent : pre.textContent;
      navigator.clipboard.writeText(text).then(function () {
        btn.innerHTML =
          '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">' +
          '<polyline points="20 6 9 17 4 12"></polyline>' +
          '</svg>';
        btn.setAttribute('aria-label', 'コピーしました!');
        setTimeout(function () {
          btn.innerHTML =
            '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">' +
            '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>' +
            '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>' +
            '</svg>';
          btn.setAttribute('aria-label', 'コードをコピー');
        }, 2000);
      });
    });
  });

  // ============================================
  // Read History (localStorage)
  // ============================================
  (function () {
    var STORAGE_KEY = 'ai-native-read-posts';

    function getReadPosts() {
      try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      } catch (e) {
        return {};
      }
    }

    function saveReadPost(url, date) {
      var posts = getReadPosts();
      posts[url] = date;
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(posts));
      } catch (e) {}
    }

    function getLatestReadDate(readPosts) {
      var latest = null;
      Object.keys(readPosts).forEach(function (url) {
        var d = new Date(readPosts[url]);
        if (!latest || d > latest) {
          latest = d;
        }
      });
      return latest;
    }

    function createBadge(type) {
      var badge = document.createElement('span');
      badge.className = 'post-status-badge post-status-badge--' + type;
      badge.textContent = type === 'read' ? '既読' : '新着';
      return badge;
    }

    function getPostStatus(url, date, readPosts, latestDate) {
      if (readPosts[url]) return 'read';
      if (latestDate && new Date(date) > latestDate) return 'new';
      return null;
    }

    // Record current post as read
    var postArticle = document.querySelector('article.post[data-post-url]');
    if (postArticle) {
      var postUrl = postArticle.getAttribute('data-post-url');
      var postDate = postArticle.getAttribute('data-post-date');
      if (postUrl && postDate) {
        saveReadPost(postUrl, postDate);
      }
    }

    var readPosts = getReadPosts();
    var latestReadDate = getLatestReadDate(readPosts);

    // Apply badges to post cards (home page)
    document.querySelectorAll('.post-card[data-post-url]').forEach(function (card) {
      var url = card.getAttribute('data-post-url');
      var date = card.getAttribute('data-post-date');
      var cardBody = card.querySelector('.post-card-body');
      if (!cardBody) return;

      var status = getPostStatus(url, date, readPosts, latestReadDate);
      if (status) {
        card.classList.add('post-card--' + status);
        cardBody.insertBefore(createBadge(status), cardBody.firstChild);
      }
    });

    // Apply badges to archive list items
    document.querySelectorAll('.archive-post-item[data-post-url]').forEach(function (item) {
      var url = item.getAttribute('data-post-url');
      var date = item.getAttribute('data-post-date');

      var status = getPostStatus(url, date, readPosts, latestReadDate);
      if (status) {
        item.classList.add('archive-post-item--' + status);
        item.appendChild(createBadge(status));
      }
    });
  }());

  // ============================================
  // Scroll Reveal Animations
  // ============================================
  (function () {
    var revealElements = document.querySelectorAll(
      '.topic-card, .post-card, .section-title, .hero-badge, .hero-title, .hero-description, .hero-actions'
    );

    if (!revealElements.length || !window.IntersectionObserver) return;

    // Add reveal class and stagger delays to grid items
    revealElements.forEach(function (el) {
      el.classList.add('reveal');
    });

    // Add staggered delays to cards within grids
    document.querySelectorAll('.topics-grid .topic-card').forEach(function (card, i) {
      card.classList.add('reveal-delay-' + Math.min(i + 1, 6));
    });

    document.querySelectorAll('.posts-grid .post-card').forEach(function (card, i) {
      card.classList.add('reveal-delay-' + Math.min(i + 1, 6));
    });

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
    );

    revealElements.forEach(function (el) {
      observer.observe(el);
    });
  }());

})();
