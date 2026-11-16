#!/usr/bin/env bash
# toat-study 배포: main + gh-pages push → GitHub Pages 빌드 확인 → 라이브 확인
# 사용법: bash deploy.sh [owner/repo]   (기본값 onedud/toat-study)
#
# 참고: Pickle의 GitHub App 권한에는 Pages 관리(pages:write)가 없어서
#       POST/PUT /repos/{owner}/{repo}/pages 는 403 "Resource not accessible by integration"이 됩니다.
#       Pages가 꺼진 저장소라도 gh-pages 브랜치를 push하면 GitHub이 자동으로 Pages를 켜고
#       source=gh-pages / /(root) 로 게시합니다(실측 확인). 그래서 main 내용을 gh-pages에도 같이 밀어
#       두 브랜치를 항상 동일하게 유지합니다.
set -euo pipefail

REPO="${1:-onedud/toat-study}"
OWNER="${REPO%%/*}"
NAME="${REPO##*/}"
URL="https://${OWNER}.github.io/${NAME}/"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== 1. push (main + gh-pages 동일하게) =="
cd "$HERE"
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/${REPO}.git"
git push origin main:main
git push origin main:gh-pages || echo "  (gh-pages push 실패: Settings에서 Pages를 직접 켜야 합니다)"

echo "== 2. Pages 상태 =="
if ! gh api "repos/${REPO}/pages" --jq '"  source: \(.source.branch)\(.source.path)  build: \(.status)"' 2>/dev/null; then
  echo "  Pages가 아직 꺼져 있습니다. GitHub → Settings → Pages → Deploy from a branch / main / / (root) → Save"
  exit 1
fi

echo "== 3. build status =="
for i in $(seq 1 20); do
  status=$(gh api "repos/${REPO}/pages/builds/latest" --jq .status 2>/dev/null || echo unknown)
  echo "  build: ${status}"
  [ "$status" = "built" ] && break
  sleep 10
done

echo "== 4. live URL =="
for i in $(seq 1 20); do
  code=$(curl -s -o /tmp/toat_live.html -w '%{http_code}' "$URL" || echo 000)
  echo "  HTTP ${code}"
  if [ "$code" = "200" ] && grep -q 'Tody' /tmp/toat_live.html; then
    echo "OK: $URL"
    exit 0
  fi
  sleep 10
done
echo "FAILED to verify $URL"
exit 1
