#!/bin/bash

# ligand_search 디렉토리로 이동
cd ligand_search

# Docker Compose 실행 (백그라운드 모드)
docker-compose up -d

# 서버가 준비될 때까지 대기
echo "Waiting for the server to be ready..."

while ! curl -s http://localhost:8000 > /dev/null; do
    echo "Server not ready yet..."
    sleep 1
done

# 서버가 준비되면 브라우저 열기
echo "Server is up, opening browser..."
xdg-open http://localhost:8000/  # 또는 open (macOS의 경우)

# Docker Compose 로그를 실시간으로 출력
docker-compose logs -f
