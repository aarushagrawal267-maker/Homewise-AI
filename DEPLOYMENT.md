# Deploying HomeWise AI

The app is deployment-ready:

* **Trained models are committed** in `models/`, so no training happens on the server.
* **The dataset downloads itself on first launch.** The Kaggle CSVs are not stored in the repo (Kaggle's rules), so
  on a fresh server the app fetches the checksum-verified files once, which takes a few seconds.
* **The 3D engine is served locally** from `static/` with a CDN fallback.
* **Dependencies are pinned**, so the saved models load exactly as trained.

Pick one of the options below.

---

## Option A — Streamlit Community Cloud (free, recommended)

1. **Make sure the code is on GitHub.** It already is: `aarushagrawal267-maker/homewise-ai`. The finished code is
   on branch `claude/project-completion`; merge it into `main` first (open a pull request on GitHub and click
   **Merge**) so that `main` has the latest version.
2. **Sign in.** Go to <https://share.streamlit.io> and sign in with GitHub. For a private repository, allow
   Streamlit to access it when asked.
3. **Create the app.** Click **Create app**, then **Deploy a public app from GitHub**, and fill in:
   * Repository: `aarushagrawal267-maker/homewise-ai`
   * Branch: `main` (after merging) or `claude/project-completion`
   * Main file path: `app.py`
   * App URL: choose a name, e.g. `homewise-ai`
4. **Choose Python 3.12.** Open **Advanced settings** and set Python version **3.12**.
5. **Deploy.** Click **Deploy**. The first build installs the requirements (about 3–5 minutes). The first visit then
   downloads the dataset and loads the model (about 20–40 seconds), and later visits are fast.

You get a public URL like `https://homewise-ai.streamlit.app`. Every push to the branch redeploys automatically.

Notes:

* The free tier sleeps after inactivity, so the first visit after a sleep takes about 30 seconds to wake up.
* The free tier has about 1 GB of RAM. The app uses roughly 400–600 MB.
* Experiments run in the AI Experiment Lab are written to the server's disk and are lost when the app restarts. That
  is expected on a free host.

## Option B — Hugging Face Spaces (free, Docker)

1. Go to <https://huggingface.co/new-space> and choose **Docker → Blank**, with free CPU hardware.
2. Push this repository to the Space. Hugging Face rejects pushes whose history contains binary files
   (`models/*.joblib`, `*.png`) that are not stored with Git LFS, and a new Space already has a first commit, so
   push a one-commit snapshot with Git LFS (this changes neither `main` nor GitHub):
   ```bash
   git lfs install
   git remote add space https://huggingface.co/spaces/<your-username>/homewise-ai
   git checkout main                       # or claude/project-completion if not merged yet
   git checkout --orphan hf-space          # new branch with no history, same files
   git lfs track "*.joblib" "*.png"
   git add .gitattributes && git add --renormalize . && git commit -m "HomeWise AI for Spaces"
   git push space hf-space:main --force    # overwrites only the Space's own repository
   git checkout main
   ```
   Alternatively, create the Space from the web UI and upload the repository files there.
3. Add this header to the top of `README.md` **in the Space** so it uses port 8501:
   ```
   ---
   title: HomeWise AI
   sdk: docker
   app_port: 8501
   ---
   ```
   Hugging Face may also require a write token for pushing, and Spaces limit file sizes; all files here are under
   10 MB.

## Option C — Render, Railway, Fly.io or your own server (Docker)

The included `Dockerfile` builds a production image: Python 3.12, a non-root user, a health check, the dataset
baked in at build time, and the `$PORT` environment variable honoured.

```bash
docker build -t homewise-ai .
docker run -p 8501:8501 homewise-ai
```

Then open <http://localhost:8501>. On Render, create a **Web Service** from the GitHub repo and choose **Docker**;
Render sets `$PORT` automatically.

## Option D — Run it on your Windows laptop

See the README, sections 19–22.

---

## After deploying, check

* **Home** shows the rotating 3D house with a live price, and the live model dashboard shows R² 0.942.
* **Property Valuation:** pressing **Estimate value** returns a price, a range and a 3D house.
* **About** shows ✓ for train.csv and test.csv, which confirms the automatic download worked.

If the dataset download is blocked on your host, upload `train.csv` and `test.csv` to `data/raw/`; the app picks
them up automatically.
