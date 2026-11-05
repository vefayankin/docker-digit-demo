# Live demo

## 1. Build 
Show the Dockerfile on GitHub for a few seconds, then:
```bash
docker build -t digit-api:1.0 .
```

## 2. Same environment everywhere 
```bash
python3 -c "import sklearn"                       
docker run --rm digit-api:1.0 python -c "import sys, sklearn; print(sys.version.split()[0], sklearn.__version__)"
```

## 3. Run and ask the model 
```bash
docker run -d --name api -p 8000:8000 digit-api:1.0
docker ps                                          
curl localhost:8000/sample/12                      
curl -s localhost:8000/info | python3 -m json.tool
```

## 4. Retrain with Compose 
```bash
docker rm -f api
docker compose up -d
docker compose run --rm trainer 
docker compose restart api
curl localhost:8000/sample/12     
```

## 5. CI/CD on GitHub 
Actions tab: test stage → smoke test → push to GHCR. Packages page: the published image.

## If something breaks
- `port is already allocated`: `docker rm -f api; docker compose down`, then repeat the command.

