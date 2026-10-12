# Local Google Gemma + Hugging Face Setup

This project can run without an OpenAI API call by using Google's Gemma 3 1B
instruction-tuned model from Hugging Face.

Model used by default:

`google/gemma-3-1b-it`

Hugging Face's model card provides a Transformers example for this checkpoint
and lists its model files at roughly 2 GB. Gemma repositories require agreeing
to Google's Gemma usage terms before model files can be downloaded.

## Step 1 — Create/verify a Hugging Face account

Go to Hugging Face and sign in.

Open the `google/gemma-3-1b-it` model page and accept the Gemma usage conditions
if prompted.

## Step 2 — Create a Hugging Face access token

In Hugging Face account settings, create a token with permission sufficient
to download public/gated model files.

Keep this token private.

## Step 3 — Put the token in `.env`

Open the project's `.env` file.

Use:

```text
MODEL_PROVIDER=local
HF_MODEL_ID=google/gemma-3-1b-it
HF_TOKEN=YOUR_HUGGING_FACE_TOKEN
```

You do NOT need an OpenAI API key when `MODEL_PROVIDER=local`.

The token is used only for downloading/accessing the Hugging Face model.

## Step 4 — Install the local-model packages

Open the VS Code terminal in the project folder:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Step 5 — Optional: log in using the Hugging Face CLI

Instead of keeping `HF_TOKEN` in `.env`, you can authenticate once:

```powershell
hf auth login
```

Paste your Hugging Face token when asked.

If you do this, you can leave:

```text
HF_TOKEN=
```

in `.env`.

## Step 6 — Start the project

Run:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The first question can take longer because the model is downloaded and loaded.

After it is loaded, the backend keeps the model in memory while the server runs.

## Step 7 — Confirm that Gemma is being used

The top-right status in the web interface should say something similar to:

```text
Local Gemma ready · local:google/gemma-3-1b-it
```

The API health endpoint is:

```text
http://127.0.0.1:8000/api/health
```

It should report:

```json
{
  "provider": "local",
  "local_model_loaded": true
}
```

after the first local generation.

## Step 8 — Test the teaching agent

Ask:

```text
Explain a buck converter from the basics.
```

Then try:

```text
Give me a numerical problem on a buck converter.
```

Then:

```text
Quiz me on PWM.
```

The local model receives the Power Electronics context from:

```text
context/power_electronics_context.md
```

so it is not being used as an unrestricted generic chatbot.

## Important laptop note

`google/gemma-3-1b-it` is intentionally used as the default local model because it is much more practical for a normal laptop than a 12B/27B-class model.

If your computer has a capable NVIDIA GPU, you can experiment with:

```text
HF_MODEL_ID=google/gemma-3-4b-it
```

but a larger model requires substantially more memory and may run slowly or fail on a laptop without enough VRAM/RAM.

## If you want OpenAI instead

Change:

```text
MODEL_PROVIDER=local
```

to:

```text
MODEL_PROVIDER=openai
```

and set:

```text
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-5-mini
```

Then restart the server.

## Switching between the two

### Local Gemma

```text
MODEL_PROVIDER=local
HF_MODEL_ID=google/gemma-3-1b-it
```

### OpenAI

```text
MODEL_PROVIDER=openai
OPENAI_MODEL=gpt-5-mini
```

Restart FastAPI after changing `.env`.

## Common errors

### Access/gated model error

Make sure you accepted Google's Gemma terms on Hugging Face and that your Hugging Face authentication is valid.

### `No module named transformers`

Run:

```powershell
pip install -r requirements.txt
```

### `No module named torch`

Run:

```powershell
pip install -r requirements.txt
```

### Out of memory

Use the 1B model:

```text
HF_MODEL_ID=google/gemma-3-1b-it
```

and close memory-heavy applications.

### Very slow CPU generation

This is expected for a local language model on CPU. A supported GPU can make local inference much faster.

### PowerShell activation error

Use:

```cmd
.venv\Scripts\activate.bat
```

from Command Prompt instead of PowerShell.

## Security

Do not publish your Hugging Face token.

Do not commit `.env` to GitHub.

The project's `.gitignore` already excludes `.env`.
