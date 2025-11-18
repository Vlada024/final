# Shape Counter Trainer — Vite React Project

This repository contains a minimal Vite React project structure that includes your provided `DotCounterTrainer` React component unchanged. It's prepared so you can build and publish the site to GitHub Pages.

## What I added

- `package.json` with scripts for `dev`, `build`, `preview`, and `deploy` (uses `gh-pages`).
- `vite.config.js` configured with `base: '/final/'` (update if your repo name differs).
- `index.html` that mounts the app and includes the Tailwind Play CDN for the UI classes used in the component.
- `src/main.jsx` to render the component.
- `src/components/DotCounterTrainer.jsx` — your component (unchanged except for fixing one accidental paste error).
- `src/styles.css` minimal stylesheet.
- `.gitignore`.

## How to run locally

1. Install dependencies:

```bash
npm install
```

2. Start dev server:

```bash
npm run dev
```

Open `http://localhost:5173` (or the URL that Vite prints).

## Build and publish to GitHub Pages

1. Build the static site:

```bash
npm run build
```

2. Deploy to GitHub Pages (this uses `gh-pages` and will push the `dist` directory to the `gh-pages` branch):

```bash
npm run deploy
```

Make sure the `homepage` in `package.json` matches `https://<your-user>.github.io/<repo>/`.

If you prefer to publish from the `docs/` folder, change the `build` output or copy the `dist` contents into `docs/` and push.

## Notes

- I did not modify your component logic; I only placed it inside `src/components/DotCounterTrainer.jsx` so you can build and publish the app.
- The Vite `base` is currently set to `/final/` (the repository name). Update `vite.config.js` if you publish under a different repo name, or remove the base if publishing to a user/organization root site.

If you want, I can:
- Run `npm install` and `npm run build` here (requires network), or
- Adjust the `base` automatically to a different repo name, or
- Add a GitHub Actions workflow to automatically deploy on push to `main`.

Which of those would you like next?