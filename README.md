# Odunayo Ogundepo

A writing-first personal publication for technical essays on language-model
systems, inference, evaluation, and research in practice.

## Local development

```bash
npm install
npm run dev
```

Run validation and create the static site:

```bash
npm run check
npm run build
```

The production output is written to `dist/`.

## Content

Launch articles live in `src/content/writing/` as Markdown files with validated
front matter. Published content automatically appears in the writing archive,
topic pages, RSS feed, and sitemap.

The previous site remains in `docs/` as a rollback source until the replacement
has completed local review and publication is explicitly approved.

## Publishing

GitHub Pages hosts the static Astro build. Pull requests run the build-only CI
workflow. Pushes to `master`, or a manual deployment run, validate the site,
upload `dist/` as a Pages artifact, and deploy it through GitHub Actions.

The pre-redesign site remains available in Git history and through the release
rollback tag.
