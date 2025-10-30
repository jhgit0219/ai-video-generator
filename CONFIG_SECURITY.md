# Configuration Security Guide

## Setup Instructions

1. **Copy the example config:**

   ```bash
   cp config.example.py config.py
   ```

2. **Add your credentials to `config.py`:**

   - Proxy credentials (if using proxies)
   - API keys (if any are added in the future)
   - Any other sensitive configuration

3. **Never commit `config.py` to version control**
   - `config.py` is already listed in `.gitignore`
   - Only commit `config.example.py` with placeholder values

## Sensitive Configuration Items

### Proxy Settings (Lines 68-80 in config.py)

If you're using a proxy service:

```python
USE_PROXIES = True
ROTATING_PROXY_ENDPOINT = "http://username:password@proxy.example.com:80"
```

**Format:** `http://username:password@host:port`

**Example (Webshare.io):**

```python
ROTATING_PROXY_ENDPOINT = "http://your-user-rotate:your-pass@p.webshare.io:80"
```

### Multiple Proxies

```python
PROXY_LIST = [
    "http://user1:pass1@proxy1.example.com:8080",
    "http://user2:pass2@proxy2.example.com:8080",
]
PROXY_ROTATION_MODE = "round_robin"  # or "random"
```

## Migration to .env (Recommended)

For better security, migrate sensitive values to a `.env` file:

1. **Install python-dotenv:**

   ```bash
   pip install python-dotenv
   ```

2. **Create `.env` file:**

   ```env
   # Proxy credentials
   PROXY_USERNAME=your-username
   PROXY_PASSWORD=your-password
   PROXY_HOST=proxy.example.com
   PROXY_PORT=80

   # Future API keys
   # OPENAI_API_KEY=sk-...
   # ANTHROPIC_API_KEY=sk-ant-...
   ```

3. **Update `config.py` to load from .env:**

   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()

   # Proxy settings
   PROXY_USERNAME = os.getenv("PROXY_USERNAME", "")
   PROXY_PASSWORD = os.getenv("PROXY_PASSWORD", "")
   PROXY_HOST = os.getenv("PROXY_HOST", "")
   PROXY_PORT = os.getenv("PROXY_PORT", "80")

   # Build proxy URL from env vars
   if PROXY_USERNAME and PROXY_PASSWORD and PROXY_HOST:
       ROTATING_PROXY_ENDPOINT = f"http://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_HOST}:{PROXY_PORT}"
   else:
       ROTATING_PROXY_ENDPOINT = ""
   ```

4. **Add `.env` to `.gitignore`** (already added)

## Security Checklist

- [ ] `config.py` is in `.gitignore`
- [ ] `config.example.py` has no real credentials
- [ ] `.env` file is in `.gitignore` (if using)
- [ ] Never commit files with credentials
- [ ] Rotate credentials if accidentally committed

## What's Protected

✅ **Protected (not committed):**

- `config.py` - Your actual configuration with secrets
- `.env` - Environment variables (if you migrate to this)
- `data/` - All generated data and caches

✅ **Safe to commit:**

- `config.example.py` - Template with placeholders
- All Python source code
- `requirements.txt`
- Documentation files

## If You Accidentally Commit Secrets

1. **Rotate credentials immediately** (change passwords, regenerate API keys)
2. **Remove from git history:**

   ```bash
   # Remove file from all commits
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch config.py" \
     --prune-empty --tag-name-filter cat -- --all

   # Force push (WARNING: rewrites history)
   git push origin --force --all
   ```

3. **Add to .gitignore and recommit**
