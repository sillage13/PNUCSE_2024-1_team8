#!/bin/bash

# p7zip 설치 확인 후 없으면 설치
if ! dpkg -l | grep -q p7zip-full; then
    echo "p7zip-full is not installed. Installing..."
    sudo apt update
    sudo apt install -y p7zip-full
fi

# 압축 파일 풀기
cd 2M
7z x data.7z.001

# 압축 파일 삭제
rm *.7z.*