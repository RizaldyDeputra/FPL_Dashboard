# 🚀 GitHub + Streamlit Cloud Deployment Guide

Complete step-by-step guide to deploy the FPL AI Optimizer publicly with
automatic daily data updates.

---

## Architecture overview

```
GitHub repo (your code + static CSV)
    │
    ├── GitHub Actions (runs daily at 06:00 UTC)
    │     └── Fetches FPL API → commits updated CSV → pushes to repo
    │
    └── Streamlit Community Cloud (hosts the dashboard)
          └── Reads from repo → auto-redeploys on push
```

---

## Prerequisites

- GitHub account (free)
- Streamlit Community Cloud account (free) → sign up at share.streamlit.io
- Anthropic API key (for the AI advisor tab) → console.anthropic.com

---

## Step 1 — Prepare your repository

### 1a. Create a new GitHub repo

Go to github.com → New repository → name it `fpl-ai-optimizer`
- Set to **Public** (required for free Streamlit Cloud)
- Do NOT initialise with README (you'll push your own)

### 1b. Push your project

Open a terminal in the `fpl_optimizer/` folder:

```bash
git init
git add .
git commit -m "feat: initial FPL AI Optimizer v3"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fpl-ai-optimizer.git
git push -u origin main
```

> The `.gitignore` already excludes secrets, model pickles, logs, and
> auto-generated data files. Only source code and `data/players.csv`
> (static fallback) are committed.

---

## Step 2 — Deploy on Streamlit Community Cloud

### 2a. Connect your repo

1. Go to **share.streamlit.io** and sign in with GitHub
2. Click **New app**
3. Select your repo: `YOUR_USERNAME/fpl-ai-optimizer`
4. Branch: `main`
5. Main file path: `app.py`
6. Click **Deploy**

### 2b. Add your Anthropic API key as a Secret

In your deployed app:
1. Click the **⋮** (three dots) → **Settings**
2. Open the **Secrets** tab
3. Paste exactly this (replace with your real key):

```toml
ANTHROPIC_API_KEY = "sk-ant-api03-your-actual-key-here"
```

4. Click **Save** — the app restarts automatically

> The `.streamlit/secrets.toml` file in your repo is a **template only**
> and is excluded from git via `.gitignore`. Never commit your real key.

---

## Step 3 — Set up automatic daily data updates (GitHub Actions)

The workflow file `.github/workflows/update_fpl_data.yml` is already in
your repo. It runs automatically — but you need to allow it to push commits.

### 3a. Enable Actions write permissions

In your GitHub repo:
1. Go to **Settings → Actions → General**
2. Scroll to **Workflow permissions**
3. Select **Read and write permissions**
4. Click **Save**

### 3b. Verify the workflow runs

1. Go to the **Actions** tab in your GitHub repo
2. Click **FPL Data Pipeline** on the left
3. Click **Run workflow** → **Run workflow** to trigger manually
4. Watch the run — it should complete in ~2 minutes

After a successful run you'll see a new commit on `main`:
```
chore: auto-update FPL data [GW29] 2025-03-15 06:02 UTC
```

Streamlit Cloud detects this push and automatically redeploys with the
fresh data.

---

## Step 4 — Verify everything works

After deployment, your app should:

| Feature | Expected behaviour |
|---|---|
| Dashboard loads | Shows "⚪ Static CSV" initially, then fetches live data on first "Refresh" click |
| Refresh button | Calls FPL API, updates badge to "🟢 Live · Xm ago" |
| AI Advisor | If API key is set in secrets, responds to chat messages |
| Data badge | Shows "🟢 Live" after first successful API fetch |
| Pipeline tab | Shows file sizes, last fetch time, automation commands |
| GitHub Actions | Commits updated CSV daily at 06:00 UTC |

---

## Troubleshooting

### App fails to start with "Data unavailable"

The static CSV (`data/players.csv`) is missing from the repo.
Fix: ensure it was committed — `git add data/players.csv && git commit --amend --no-edit && git push --force`

### AI Advisor shows "not configured"

Your secret is not set correctly. Double-check:
- Key name must be exactly `ANTHROPIC_API_KEY` (case-sensitive)
- Value must include `sk-ant-` prefix
- Saved in the Secrets tab (not a `.env` file)

### GitHub Actions fails with "Permission denied" on push

Workflow permissions are not set to read+write. See Step 3a above.

### Data is stale / "🟡 Cached" badge

The GitHub Action hasn't run yet, or the FPL API was temporarily unavailable.
Click **Run workflow** manually in the Actions tab to force a refresh.

### "Module not found" on Streamlit Cloud

All dependencies must be in `requirements.txt`. Check the cloud logs for the
exact missing module and add it.

---

## Managing costs

| Service | Cost |
|---|---|
| GitHub repo (public) | Free |
| GitHub Actions | Free (2,000 min/month on free tier) |
| Streamlit Community Cloud | Free (1 app, public repo) |
| Anthropic API | Pay-per-use (~$0.003 per AI chat message) |
| FPL API | Free (official public API) |

GitHub Actions usage: each daily run takes ~90 seconds = ~45 min/month,
well within the free 2,000 minute limit.

---

## Optional enhancements

### Custom domain on Streamlit Cloud

Settings → General → Custom subdomain → set `fpl-optimizer` for URL:
`fpl-optimizer.streamlit.app`

### Protect with a password

Add to `.streamlit/secrets.toml` (Streamlit Cloud secrets tab):
```toml
APP_PASSWORD = "your-password"
```
Then add to the top of `app.py`:
```python
pwd = st.text_input("Password", type="password")
if pwd != st.secrets.get("APP_PASSWORD", ""):
    st.stop()
```

### Multiple update times

Edit `.github/workflows/update_fpl_data.yml` to add more cron entries:
```yaml
schedule:
  - cron: "0 6 * * *"    # 06:00 UTC daily
  - cron: "0 18 * * 4"   # 18:00 UTC Thursdays (deadline day cover)
```

---

## File summary for deployment

```
fpl_optimizer/
├── app.py                              ← main Streamlit app
├── cloud_startup.py                    ← ensures data exists at boot
├── update_data.py                      ← data pipeline (used by Actions)
├── requirements.txt                    ← Python dependencies
├── packages.txt                        ← system packages (currently empty)
├── .gitignore                          ← excludes secrets, cache, logs
├── .streamlit/
│   ├── config.toml                     ← theme + server settings
│   └── secrets.toml                    ← TEMPLATE ONLY — never commit real values
├── .github/
│   └── workflows/
│       └── update_fpl_data.yml         ← daily GitHub Actions pipeline
└── data/
    └── players.csv                     ← static fallback (committed to git)
```
