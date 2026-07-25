# Blog Posts Directory

This directory contains individual blog post HTML files.

## Adding a New Blog Post

To add a new blog post:

1. **Create a new HTML file** in this directory (e.g., `my-new-post.html`)
   - Use `_template.html` as a template
   - Update the title, meta description, and content
   - Make sure all asset links use `../` to go up one directory (e.g., `../assets/`, `../styles.css`)

2. **Add the post to the blog listing** in `/docs/blog.html`
   - Copy one of the existing `<article class="blog-post-card">` blocks
   - Update the title, link, date, tags, and excerpt
   - The link should point to `posts/your-filename.html`

3. **File naming convention**
   - Use lowercase with hyphens: `my-blog-post-title.html`
   - Keep it concise but descriptive

## Blog Post Template Structure

```html
<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Post Title - Odunayo Ogundepo</title>
    <meta name="description" content="Brief description">
    <link rel="icon" type="image/x-icon" href="../assets/favicon.ico">
    <link rel="stylesheet" href="../styles.css">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <!-- Theme script... -->
</head>
<body>
    <!-- Header (copy from existing post) -->

    <main class="mx-auto max-w-3xl px-4 sm:px-8">
        <!-- Back Link -->
        <div class="mt-8 mb-8">
            <a href="../blog.html" class="text-accent-2 hover:text-accent transition-colors">← Back to Blog</a>
        </div>

        <!-- Article -->
        <article class="prose mb-16">
            <header class="mb-8">
                <h1 class="title text-4xl font-bold mb-4">Your Post Title</h1>
                <div class="text-muted mb-4">
                    <time datetime="YYYY-MM-DD">Month Day, Year</time>
                    <span class="mx-2">•</span>
                    <span>X min read</span>
                </div>
                <div class="flex gap-2 flex-wrap">
                    <span class="tag tag-safety">Tag1</span>
                    <span class="tag tag-arxiv">Tag2</span>
                </div>
            </header>

            <div class="blog-content">
                <!-- Your content here -->
            </div>
        </article>
    </main>

    <!-- Footer -->
    <script src="../script.js"></script>
</body>
</html>
```

## Available CSS Classes

- `.prose` - Blog content styling
- `.title` - Section titles
- `.tag` with modifiers (`tag-safety`, `tag-arxiv`, `tag-github`) - Colored tags
- `.cactus-link` - Styled links
- `.text-muted` - Muted text color
- `.lead` - Larger intro paragraph

## Tips

- Keep blog posts focused and concise
- Use proper semantic HTML (h2, h3, p, ul, etc.)
- Add code examples using `<pre><code>` blocks
- Include relevant links and references
- Update the date format to be consistent: `<time datetime="2026-01-15">January 15, 2026</time>`
