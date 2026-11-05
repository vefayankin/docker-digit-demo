# Docker for MLOps: a handwritten-digit recognizer in a container

A small neural network (scikit-learn `MLPClassifier`) that recognises handwritten
digits from 8x8 images, served with FastAPI and packaged with Docker.
The image contains **code + Python + libraries + trained model**, so it behaves
the same on a laptop, on GitHub Actions and on a server.

Dataset: scikit-learn's bundled `digits` dataset (1,797 images of digits 0–9), no download needed.

```
docker-digit-demo/
├── app/
│   ├── train.py          # trains the model (mlp or rf), saves model + metadata
│   └── main.py           # FastAPI: /health, /info, /predict, /sample/{i}
├── tests/test_api.py     # runs inside the image; accuracy gate >= 0.95
├── examples/digit.json   # a "3" as 64 pixel values
├── Dockerfile            # base → deps → train → serve, plus test
├── docker-compose.yml    # api + trainer sharing a "models" volume
├── DEMO.md               # live demo script
└── .github/workflows/docker.yml   # CI: test → smoke test → push to GHCR
```

## Run it

Requires Docker Desktop (or Docker Engine with Compose v2).

```bash
docker build -t digit-api:1.0 .                      # model is trained during the build
docker run -d --name api -p 8000:8000 digit-api:1.0
curl localhost:8000/sample/12                        # draws a test digit and classifies it
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @examples/digit.json
```

Swagger UI: http://localhost:8000/docs

## Endpoints

| Method | Path          | Returns |
|--------|---------------|---------|
| GET    | `/health`     | `{"status": "ok"}` (used by the Docker `HEALTHCHECK`) |
| GET    | `/info`       | container hostname, Python version, model metadata |
| POST   | `/predict`    | body `{"pixels": [64 values 0–16]}` → digit + probability |
| GET    | `/sample/{i}` | plain text: test image `i` drawn in ASCII, true and predicted digit |

## Dockerfile stages

| Stage   | Purpose |
|---------|---------|
| `base`  | `python:3.12-slim`, environment variables, workdir |
| `deps`  | installs pinned `requirements.txt` (cached layer) |
| `train` | trains the model inside the build (`--build-arg MODEL_TYPE=rf` to change it) |
| `serve` | final image: app + model, non-root user, health check |
| `test`  | runs `pytest` on top of `serve`; a failing test fails the build |

## Retrain without rebuilding

```bash
docker compose up -d               # API starts; the volume is filled from the image
docker compose run --rm trainer    # trains a random forest into the shared volume
docker compose restart api         # API loads the new model
curl localhost:8000/info           # model_type is now "rf"
docker compose down -v             # stop everything and delete the volume
```

## CI/CD

`.github/workflows/docker.yml` runs on every push and pull request:
1. builds the `test` stage (tests inside the image),
2. builds the `serve` image, starts it and calls `/health`, `/sample/0` and `/predict`,
3. on `main`, pushes the image to `ghcr.io/<owner>/<repo>:latest` and `:<commit-sha>`.

New GHCR packages are private: set the package to **Public** in its settings so that anyone can run
`docker run -p 8000:8000 ghcr.io/<owner>/<repo>:latest`.
