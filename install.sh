#!/bin/sh
# sage installer — downloads pre-built binary from GitHub Releases
# Usage: curl -fsSL https://raw.githubusercontent.com/MisterTK/sage/main/install.sh | sh
set -eu

REPO="MisterTK/sage"

info() { printf '  \033[1;34m%s\033[0m %s\n' "$1" "$2"; }
err()  { printf '  \033[1;31merror:\033[0m %s\n' "$1" >&2; exit 1; }

# Detect OS
case "$(uname -s)" in
  Darwin) os="apple-darwin" ;;
  Linux)  os="unknown-linux-gnu" ;;
  *)      err "Unsupported OS: $(uname -s). Download manually from https://github.com/${REPO}/releases" ;;
esac

# Detect architecture
case "$(uname -m)" in
  x86_64|amd64)  arch="x86_64" ;;
  arm64|aarch64) arch="aarch64" ;;
  *)             err "Unsupported architecture: $(uname -m)" ;;
esac

target="${arch}-${os}"
info "Platform" "${target}"

# Determine latest version
if [ -n "${SAGE_VERSION:-}" ]; then
  version="$SAGE_VERSION"
else
  info "Fetching" "latest release..."
  version=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
    | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')
  [ -z "$version" ] && err "Could not determine latest version. Set SAGE_VERSION=v0.1.0 manually."
fi
info "Version" "${version}"

# Download
archive="sage-${version}-${target}.tar.gz"
url="https://github.com/${REPO}/releases/download/${version}/${archive}"
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

info "Downloading" "${url}"
curl -fSL --progress-bar -o "${tmpdir}/${archive}" "$url" \
  || err "Download failed. Check https://github.com/${REPO}/releases for available builds."

# Verify checksum if available
checksum_url="${url}.sha256"
if curl -fsSL -o "${tmpdir}/${archive}.sha256" "$checksum_url" 2>/dev/null; then
  expected=$(awk '{print $1}' "${tmpdir}/${archive}.sha256")
  if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "${tmpdir}/${archive}" | awk '{print $1}')
  elif command -v shasum >/dev/null 2>&1; then
    actual=$(shasum -a 256 "${tmpdir}/${archive}" | awk '{print $1}')
  else
    actual=""
  fi
  if [ -n "$actual" ] && [ "$actual" != "$expected" ]; then
    err "Checksum mismatch! Expected ${expected}, got ${actual}"
  fi
  [ -n "$actual" ] && info "Checksum" "verified"
fi

# Extract binary
tar xzf "${tmpdir}/${archive}" -C "$tmpdir"
bin_src="${tmpdir}/sage-${version}-${target}/sage"

# Copy ONNX Runtime dylib alongside binary (needed for model inference)
dylib_src=""
for lib in "${tmpdir}/sage-${version}-${target}"/libonnxruntime*; do
  [ -f "$lib" ] && dylib_src="$lib" && break
done

# Pick install dir — prefer /usr/local/bin (no PATH change needed), fallback to ~/.local/bin
if [ -w /usr/local/bin ] || sudo -n true 2>/dev/null; then
  INSTALL_DIR="/usr/local/bin"
  USE_SUDO=""
  [ -w /usr/local/bin ] || USE_SUDO="sudo"
else
  INSTALL_DIR="${HOME}/.local/bin"
  USE_SUDO=""
fi

info "Installing" "${INSTALL_DIR}/sage"
${USE_SUDO} mkdir -p "$INSTALL_DIR"
${USE_SUDO} cp "$bin_src" "${INSTALL_DIR}/sage"
${USE_SUDO} chmod +x "${INSTALL_DIR}/sage"

# Copy dylib next to binary if present
if [ -n "$dylib_src" ]; then
  ${USE_SUDO} cp "$dylib_src" "$INSTALL_DIR/"
fi

# Ensure INSTALL_DIR is in PATH (only needed for ~/.local/bin fallback)
if [ "$INSTALL_DIR" != "/usr/local/bin" ]; then
  shell_rc=""
  if [ -f "$HOME/.zshrc" ]; then
    shell_rc="$HOME/.zshrc"
  elif [ -f "$HOME/.bashrc" ]; then
    shell_rc="$HOME/.bashrc"
  elif [ -f "$HOME/.profile" ]; then
    shell_rc="$HOME/.profile"
  fi

  if [ -n "$shell_rc" ] && ! grep -q "${INSTALL_DIR}" "$shell_rc" 2>/dev/null; then
    printf '\nexport PATH="%s:$PATH"\n' "$INSTALL_DIR" >> "$shell_rc"
    info "Updated" "$shell_rc"
  fi

  export PATH="${INSTALL_DIR}:$PATH"
fi

echo ""
info "Done!" "sage ${version} is ready"
echo ""
sage --version
echo ""
echo "  Next: install into your AI coding tool:"
echo "    sage install-claude-code   # Claude Code"
echo "    sage install-codex         # Codex CLI"
echo "    sage install-open-code     # OpenCode"
echo ""
