# JobPilot Code

Run commands from the project root:

```bash
pip install -r code/requirements.txt
python code/build_embeddings.py
streamlit run code/app.py --server.fileWatcherType none
python code/evaluate_personas.py
```

See the root `README.md` for full setup, data, deployment, and submission notes.
