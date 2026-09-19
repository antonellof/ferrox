#!/usr/bin/env bash
# Install frink + frink-server from a GitHub release.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/antonellof/frink/main/scripts/install.sh | bash
#
# Env:
#   FRINK_VERSION      tag to install (default: latest release)
#   FRINK_INSTALL_DIR  install directory (default: ~/.local/bin)
#   FRINK_REPO         owner/repo (default: antonellof/frink)
set -euo pipefail

REPO="${FRINK_REPO:-antonellof/frink}"
INSTALL_DIR="${FRINK_INSTALL_DIR:-${HOME}/.local/bin}"
API="https://api.github.com/repos/${REPO}"
DOWNLOAD_BASE="https://github.com/${REPO}/releases/download"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: need '$1' on PATH" >&2
    exit 1
  }
}

need curl
need tar
need uname
need mktemp

os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"

case "${os}" in
  darwin) platform="darwin" ;;
  linux) platform="linux" ;;
  *)
    echo "error: unsupported OS '${os}' (need darwin or linux)" >&2
    exit 1
    ;;
esac

case "${arch}" in
  x86_64 | amd64) arch="x86_64" ;;
  aarch64 | arm64) arch="arm64" ;;
  *)
    echo "error: unsupported arch '${arch}'" >&2
    exit 1
    ;;
esac

# Prebuilt matrix today: darwin-arm64 (Metal) + linux-x86_64 (CPU).
case "${platform}-${arch}" in
  darwin-arm64 | linux-x86_64) ;;
  *)
    echo "error: no prebuilt binary for ${platform}-${arch}" >&2
    echo "Build from source: https://github.com/${REPO}#build-from-source" >&2
    exit 1
    ;;
esac

if [[ -n "${FRINK_VERSION:-}" ]]; then
  tag="${FRINK_VERSION}"
else
  tag="$(
    curl -fsSL "${API}/releases/latest" |
      sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' |
      head -n 1
  )"
  if [[ -z "${tag}" ]]; then
    echo "error: could not resolve latest release tag for ${REPO}" >&2
    exit 1
  fi
fi

asset="frink-${tag}-${platform}-${arch}.tar.gz"
url="${DOWNLOAD_BASE}/${tag}/${asset}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

echo "Downloading ${asset}..."
if ! curl -fsSL -o "${tmpdir}/${asset}" "${url}"; then
  echo "error: failed to download ${url}" >&2
  echo "Check that release ${tag} exists and includes ${asset}." >&2
  exit 1
fi

tar -xzf "${tmpdir}/${asset}" -C "${tmpdir}"

mkdir -p "${INSTALL_DIR}"
for bin in frink frink-server; do
  src=""
  for candidate in \
    "${tmpdir}/${bin}" \
    "${tmpdir}/frink-${tag}-${platform}-${arch}/${bin}"; do
    if [[ -f "${candidate}" ]]; then
      src="${candidate}"
      break
    fi
  done
  if [[ -z "${src}" ]]; then
    echo "error: archive missing binary '${bin}'" >&2
    exit 1
  fi
  chmod +x "${src}"
  install -m 755 "${src}" "${INSTALL_DIR}/${bin}"
  echo "Installed ${INSTALL_DIR}/${bin}"
done

case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *)
    echo
    echo "Note: ${INSTALL_DIR} is not on your PATH."
    echo "Add it, for example:"
    echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
    ;;
esac

echo
echo "Done. Try: frink --help"
