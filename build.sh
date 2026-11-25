#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="${ROOT_DIR}/app"
APP_NAME="SwiftRVC"
APP_BUNDLE="${ROOT_DIR}/${APP_NAME}.app"
CONTENTS_DIR="${APP_BUNDLE}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
	
  
echo "building ${APP_NAME}..."
cd "${APP_DIR}"
swift build -c release -Xswiftc -O
	
  
rm -rf "${APP_BUNDLE}"
mkdir -p "${MACOS_DIR}" "${RESOURCES_DIR}"

cp "${APP_DIR}/.build/release/${APP_NAME}" "${MACOS_DIR}/${APP_NAME}"
cp "${APP_DIR}/Resources/Info.plist" "${CONTENTS_DIR}/Info.plist"
chmod +x "${MACOS_DIR}/${APP_NAME}"

# ad-hoc sign so macos keeps mic/audio tcc permissions stable
codesign --force --deep --sign - "${APP_BUNDLE}" >/dev/null 2>&1 || true

ls -lh "${MACOS_DIR}/${APP_NAME}"
echo "done: ${APP_BUNDLE}"
