# Gradio + Streamlit: what to do next

## 1. Add the files to your repo
Extract the zip **into the repo root** (`BI-ServingMLModels-C1/`). You get four new things; nothing existing is changed (`api.py` stays untouched):

```
inference.py          shared code: loads models/*.joblib, same logic as api.py
make_space.py         builds the folder you upload to Hugging Face
gradio_space/         app.py, requirements.txt, README.md  (Hugging Face)
streamlit_app/        app.py, requirements.txt             (Streamlit Cloud)
```

## 2. Test locally (from the repo root, with .venv active)
```bash
pip install gradio streamlit
python inference.py                      # prints each form + one test prediction per model
python gradio_space/app.py               # open http://127.0.0.1:7860
streamlit run streamlit_app/app.py       # opens http://localhost:8501
```
Do NOT add gradio/streamlit to the root `requirements.txt` (it would bloat the Docker/Render build).
Click Predict on all 4 tabs of each app. Send me the `python inference.py` output if anything looks off.

## 3. Commit on a branch (this is your documented contribution + branch link)
```bash
git checkout -b feature/gradio-streamlit
git add inference.py make_space.py gradio_space streamlit_app
git commit -m "Add Gradio and Streamlit apps"
git push -u origin feature/gradio-streamlit
```
Then open a Pull Request on GitHub and merge it into `main`.

## 4. Hugging Face Space (+4)
1. Sign in at huggingface.co, then New -> Space. Name it, pick SDK **Gradio**, hardware **CPU basic (free)**, visibility **Public**.
2. In the repo root run `python make_space.py`. It creates `hf_space/` with app.py, inference.py, requirements.txt, README.md and models/.
3. In the Space: Files -> Add file -> Upload files. Drag in **everything inside `hf_space/`** (including the `models` folder). Commit.
4. Wait for "Building" to turn into "Running" (2-5 min). Test all 4 tabs.
5. URL: `https://huggingface.co/spaces/<your-username>/<space-name>`

## 5. Streamlit Community Cloud (+4)
1. Go to share.streamlit.io, sign in with GitHub.
2. Create app -> pick repo `GraceChelangat/BI-ServingMLModels-C1`, branch `main` (after the PR is merged), **Main file path: `streamlit_app/app.py`**.
3. Advanced settings -> **Python 3.12**.
4. Deploy. URL looks like `https://<name>.streamlit.app`.
If the repo is private or belongs to Grace, she must deploy it (or give you access). Otherwise push a copy to your own public GitHub repo and deploy from that.

## 6. Fill the submission file
- Chosen level: **advanced**
- Render URL, Hugging Face URL, Streamlit URL in the Public URLs table
- Every member: Student ID, name, what they did, **branch link** (-5 marks per member left blank)
- Video link (not the video file, and check the lecturer can view it)

## 7. Video script (max 10 min, one continuous take)
| Time | Show |
|---|---|
| 0:00 | One sentence: team name, Advanced tier |
| 0:20 | `python api.py` locally. In Postman/curl, send a request to **each of the 4 endpoints** and show the response |
| 3:00 | `docker compose up`, then show the same requests going through **Nginx**. Show `docker ps` or the logs proving Gunicorn is running |
| 4:30 | Request to the **Render URL** from Postman/curl |
| 5:30 | The HTML page: one good prediction, then submit with a missing input to show the error message |
| 6:30 | **Hugging Face Space** in the browser: one prediction on all 4 tabs |
| 8:00 | **Streamlit app** in the browser: one prediction on all 4 tabs |
| 9:30 | Wrap up |

Before recording: open the Render URL and the Space a few minutes earlier (free tiers sleep and take about a minute to wake), and test every endpoint fresh. Do a dry run first, since 10 minutes is tight.
