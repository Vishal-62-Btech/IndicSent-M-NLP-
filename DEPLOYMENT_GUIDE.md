# IndicSent-M: Deployment & Operations Guide

## 1. System Requirements & Prerequisites

### 1.1 Hardware Specifications
| Component | Minimum Requirements | Recommended Production |
|:---|:---|:---|
| **Processor (CPU)** | 4 Cores, 2.5 GHz x86_64 | 8+ Cores (e.g. Intel Core i7 / AMD Ryzen 7) |
| **System Memory (RAM)** | 8 GB | 16 GB+ |
| **GPU (Hardware Acceleration)** | Optional (CPU fallback supported) | NVIDIA GPU with $\ge$ 6 GB VRAM (RTX 3060 / 4060 / 5050) |
| **GPU Compute Stack** | CUDA 11.8+ / ROCm | CUDA 12.1+ with cuDNN |
| **Disk Storage** | 10 GB free space | 25 GB free space (for checkpoints & datasets) |

### 1.2 Software Stack
- **Operating System:** Windows 10/11 (64-bit) or Linux (Ubuntu 20.04/22.04 LTS)
- **Python:** Version 3.11.x (Strictly recommended for PyTorch 2.x stability)
- **Node.js:** Version 18.x or 20.x LTS
- **Package Managers:** `pip` (Python) and `npm` (Node.js)

---

## 2. Environment Setup

### 2.1 Backend Environment Setup

1. Open PowerShell or Terminal in the project root:
   ```bash
   cd "D:\projectsss\NLP project (sentiment analysis)"
   ```

2. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - **Windows PowerShell:**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
     *(If script execution is disabled, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*
   - **Linux / macOS:**
     ```bash
     source .venv/bin/activate
     ```

4. Install backend dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r backend/requirements.txt
   ```

5. Verify PyTorch CUDA acceleration:
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
   ```

---

### 2.2 Frontend Environment Setup

1. Install Node.js dependencies from the project root:
   ```bash
   npm install
   ```

2. Verify `.env` configuration:
   Ensure `.env` in the root contains:
   ```env
   VITE_API_URL=http://127.0.0.1:8000
   ```

---

## 3. Running the System in Development

### 3.1 Step 1: Start FastAPI Backend
From the project root:
```powershell
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API will be accessible at: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`

### 3.2 Step 2: Start React Frontend
In a separate terminal window from the project root:
```powershell
npm.cmd run dev
```
- Frontend application will be live at: `http://localhost:3000` (or `http://localhost:5173`)

---

## 4. Production Build & Deployment

### 4.1 Production Frontend Build
Compile optimized static web assets:
```powershell
npm.cmd run build
```
This generates a production-ready `dist/` directory containing:
- `dist/index.html` (~1.26 KB)
- `dist/assets/index-*.css` (~28.66 KB)
- `dist/assets/index-*.js` (~252.09 KB)

To test the production build locally:
```powershell
npm.cmd run preview
```

---

### 4.2 Production Backend Deployment (Uvicorn / Gunicorn)

For production serving with PyTorch deep learning models, avoid high multi-process worker counts on a single GPU to prevent VRAM over-allocation. Use a single high-throughput asynchronous process per GPU instance:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop --http httptools
```

#### Production Systemd Service (Linux):
Create `/etc/systemd/system/indicsent-backend.service`:
```ini
[Unit]
Description=IndicSent-M FastAPI Backend Service
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/indicsent/backend
Environment="PATH=/var/www/indicsent/.venv/bin"
ExecStart=/var/www/indicsent/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable indicsent-backend
sudo systemctl start indicsent-backend
```

---

### 4.3 Nginx Reverse Proxy Configuration

To serve the React frontend and proxy `/api` calls to the FastAPI backend:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Serve React Static Files
    location / {
        root /var/www/indicsent/dist;
        index index.html index.htm;
        try_files $uri $uri/ /index.html;
    }

    # Proxy API Requests to FastAPI Backend
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 12M;
        proxy_read_timeout 60s;
        proxy_connect_timeout 60s;
    }

    # Proxy Swagger Docs
    location /docs {
        proxy_pass http://127.0.0.1:8000/docs;
    }
    location /openapi.json {
        proxy_pass http://127.0.0.1:8000/openapi.json;
    }
}
```

---

## 5. System Health Verification & Smoke Tests

After launching the backend, run the automated integration verification suite:
```powershell
python backend/tests/test_e2e_integration.py
```

Expected output:
```
--- PHASE 11 INTEGRATION VERIFICATION ---
[PASS] 1. Backend connection verified: healthy on cuda
[PASS] 2. Model catalog retrieved successfully. Modes: ['multimodal', 'text', 'image']
[PASS] 3. Text prediction verified: Positive (86.43%) in 11.2 ms
[PASS] 4. Image prediction verified: Neutral (62.14%) in 18.5 ms
[PASS] 5. Multimodal prediction verified: Negative (78.21%) in 32.1 ms
[PASS] 6. Unsupported image upload properly rejected with HTTP 415
[PASS] 7. Empty text validation properly rejected with HTTP 400
[PASS] 8. React production bundle verified
ALL INTEGRATION TESTS PASSED SUCCESSFULLY!
```

---

## 6. Troubleshooting Common Issues

### Issue 1: PowerShell Script Execution Restricted
- **Symptom:** `File ...\npm.ps1 cannot be loaded because running scripts is disabled on this system.`
- **Resolution:** Execute commands using the `.cmd` extension (e.g. `npm.cmd run dev`), or enable process-level execution:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

### Issue 2: CUDA Out of Memory (OOM)
- **Symptom:** `RuntimeError: CUDA out of memory.`
- **Resolution:**
  1. IndicSentPredictor maintains a single instance of each backbone in VRAM (~2.19 GB). Ensure background applications (e.g. games, video editors) are closed.
  2. If VRAM is below 4 GB, force CPU fallback by setting `DEVICE = "cpu"` in `backend/app/core/config.py`.

### Issue 3: Cross-Origin Resource Sharing (CORS) Blocked
- **Symptom:** `Access to XMLHttpRequest at 'http://127.0.0.1:8000/api/...' has been blocked by CORS policy.`
- **Resolution:** Add your frontend domain or origin to `CORS_ORIGINS` in `backend/app/core/config.py`.

### Issue 4: Port 8000 Already in Use
- **Symptom:** `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)`
- **Resolution:**
  Find and kill the conflicting process:
  ```powershell
  Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
  ```
