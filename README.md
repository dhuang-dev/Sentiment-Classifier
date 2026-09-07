# Sentiment-Classifier
# Sentiment Classifier — Full Stack App

A sentiment classifier with a separate backend (FastAPI) and frontend (HTML/JS),
connected over a REST API.

```
sentiment-app/
├── backend/
│   ├── main.py            # FastAPI app, loads the model, exposes /classify
│   └── requirements.txt
├── frontend/
│   └── index.html         # UI, calls the backend API
└── README.md
```

## 1. Run the backend locally

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Visit http://localhost:8000 — you should see `{"status": "ok", ...}`.
That confirms the API is alive. Test the real endpoint with:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "This was fantastic."}'
```

## 2. Run the frontend locally

Just open `frontend/index.html` directly in your browser (double-click it,
or drag it into a browser window). It's already pointed at
`http://localhost:8000/classify`, so as long as the backend is running,
typing text and clicking "Run classification" will work end to end.

## 3. Put it in git / GitHub

```bash
cd sentiment-app
git init
git add .
git commit -m "Initial commit: sentiment classifier backend + frontend"
```

Then create an empty repo on github.com (no README/gitignore, since you
already have files), and connect it:

```bash
git remote add origin https://github.com/your-username/sentiment-app.git
git branch -M main
git push -u origin main
```

## 4. Deploy it for real (optional next step)

- **Backend**: push to Render or Railway (both have free tiers) — point them
  at the `backend/` folder, they'll run `pip install -r requirements.txt`
  then `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- **Frontend**: once the backend has a public URL, change `API_URL` in
  `index.html` to that URL, then deploy the `frontend/` folder to Vercel,
  Netlify, or even GitHub Pages (all free, all just need static files).

## Notes

- The model loads once, at backend startup — the first request after
  starting the server will be a little slow while it downloads/loads
  DistilBERT; after that it's fast.
- CORS is currently wide open (`allow_origins=["*"]`) so the frontend can
  call the backend from any origin during development. Before deploying
  publicly, restrict this to your actual frontend's URL in `main.py`.

## Additionally
- Windows Powershell command: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
- Needed if user is able to run script
