# Live demo (about 3.5 minutes)

## Before the talk
- Docker Desktop running.
- Build everything once so the demo uses the cache:
  `docker build -t digit-api:1.0 .` and `docker compose --profile train build`
- Clean up: `docker rm -f api` and `docker compose down -v`
- Browser tabs: the repo, the Actions tab (a green run), the Packages page.
- Terminal in the repo folder, large font.

## 1. Build (40 s)
Show the Dockerfile on GitHub for a few seconds, then:
```bash
docker build -t digit-api:1.0 .
```
Point at: `CACHED` on the pip layer, and the accuracy printed by the train stage.

## 2. Same environment everywhere (30 s)
```bash
python3 -c "import sklearn"                        # my laptop: ModuleNotFoundError
docker run --rm digit-api:1.0 python -c "import sys, sklearn; print(sys.version.split()[0], sklearn.__version__)"
```
Point at: scikit-learn is not installed on my laptop at all; the container brings its own Python and scikit-learn.
(On Windows use `python` instead of `python3`. Check before the talk that it really fails on your machine.)

## 3. Run and ask the model (60 s)
```bash
docker run -d --name api -p 8000:8000 digit-api:1.0
docker ps                       # (healthy) after ~10 s
curl localhost:8000/sample/12   # draws a 9 and predicts it
curl localhost:8000/sample/5
```
Point at: `container:` line = the container ID. Optional: open http://localhost:8000/docs

## 4. Retrain with Compose (50 s)
```bash
docker rm -f api
docker compose up -d
docker compose run --rm trainer   # random forest into the shared volume
docker compose restart api
curl localhost:8000/sample/12     # same 9, now model: rf, lower confidence
```
Point at: new model without rebuilding the image.

## 5. CI/CD on GitHub (30 s)
Actions tab: test stage → smoke test → push to GHCR. Packages page: the published image.

## If something breaks
Show the green Actions run: it performs the same build, test and smoke test.

## Windows (PowerShell)
Use `curl.exe` instead of `curl`, and quote the file: `-d "@examples/digit.json"`.
