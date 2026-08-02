# Day 2 Deployment Record

## URLs

| Service | URL |
|---------|-----|
| Frontend | https://smart-learn-ai-gamma.vercel.app |
| Backend health | https://smartlearn-ai-production-5dea.up.railway.app/health |
| Backend docs | https://smartlearn-ai-production-5dea.up.railway.app/docs |
| Repository | https://github.com/qingyangli310/smartLearn-AI |
| Pull Request | https://github.com/qingyangli310/smartLearn-AI/pull/1 |

## Deployed Version

- **Branch**: `main`
- **Merge commit**: `6bff6a3`

## Platform Configuration

### Railway (Backend)

- **Root Directory**: `smartlearn-backend`
- **Environment variables**:
  - `OPENROUTER_API_KEY` — OpenRouter API key
  - `OPENROUTER_MODEL` — fixed model ID for citation stability
  - `ALLOWED_ORIGINS` — comma-separated CORS origins

### Vercel (Frontend)

- **Root Directory**: `smartlearn-frontend`
- **Environment variable**: `VITE_API_URL` — backend URL

## Acceptance Tests

| Test | Status | Notes |
|------|--------|-------|
| `/health` | ✅ Pass | Returns `{"ok": true}` |
| Upload PDF | ✅ Pass | sample.pdf: 11 pages, 32,613 characters |
| Known question | ✅ Pass | "attention head dimension" → 64, correctly cited Page 5 |
| Unknown question | ✅ Pass | No fabricated accuracy numbers; no phantom citations |
| Fake chat ID | ✅ Pass | Returns 404 with re-upload prompt |
| Browser Network | ✅ Pass | POST to Railway `/chat` returns 200 |
| CORS restart recovery | ✅ Pass | Re-upload after restart works correctly |

## Docker Build

- **Local**: Not tested (Docker not installed)
- **Railway**: Build passed using `smartlearn-backend/Dockerfile`

## Known Limitations

Railway restarts clear the in-memory `documents` dictionary. Users must re-upload their PDF after a server restart. A persistent storage layer (database or object store) would be required to maintain document state across deployments.
