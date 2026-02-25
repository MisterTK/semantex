#!/bin/sh
# sage installer — downloads pre-built binary from GitHub Releases
# Usage: curl -fsSL https://raw.githubusercontent.com/MisterTK/sage/main/install.sh | sh
set -eu

REPO="MisterTK/sage"
INSTALL_DIR="${SAGE_INSTALL_DIR:-$HOME/.sage/bin}"

info() { printf '  \033[1;34m%s\033[0m %s\n' "$1" "$2"; }
err()  { printf '  \033[1;31merror:\033[0m %s\n' "$1" >&2; exit 1; }

# Detect OS
case "$(uname -s)" in
  Darwin) os="apple-darwin" ;;
  Linux)  os="unknown-linux-gnu" ;;
  *)      err "Unsupported OS: $(uname -s). Use Windows builds from GitHub Releases." ;;
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
  || err "Download failed. Check that ${version} has a release for ${target}."

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

# Extract and install
info "Installing" "${INSTALL_DIR}/sage"
mkdir -p "$INSTALL_DIR"
tar xzf "${tmpdir}/${archive}" -C "$tmpdir"
# Binary is inside sage-vX.Y.Z-target/ directory
cp "${tmpdir}/sage-${version}-${target}/sage" "${INSTALL_DIR}/sage"
chmod +x "${INSTALL_DIR}/sage"

# Copy ONNX Runtime dylib if bundled
for lib in "${tmpdir}/sage-${version}-${target}"/libonnxruntime*; do
  [ -f "$lib" ] && cp "$lib" "$INSTALL_DIR/"
done

# Add to PATH if not already there
add_to_path() {
  profile="$1"
  if [ -f "$profile" ] && grep -q "${INSTALL_DIR}" "$profile" 2>/dev/null; then
    return
  fi
  printf '\n# sage\nexport PATH="%s:$PATH"\n' "$INSTALL_DIR" >> "$profile"
  info "Updated" "$profile"
}

case "${INSTALL_DIR}" in
  */bin) ;; # only update PATH if not already standard location
esac

if ! echo "$PATH" | tr ':' '\n' | grep -qx "$INSTALL_DIR"; then
  if [ -f "$HOME/.zshrc" ]; then
    add_to_path "$HOME/.zshrc"
  elif [ -f "$HOME/.bashrc" ]; then
    add_to_path "$HOME/.bashrc"
  elif [ -f "$HOME/.profile" ]; then
    add_to_path "$HOME/.profile"
  fi
fi

echo ""
info "Done!" "sage ${version} installed to ${INSTALL_DIR}/sage"
echo ""
echo "  Restart your shell or run:"
echo "    export PATH=\"${INSTALL_DIR}:\$PATH\""
echo ""
echo "  Then set up agent integrations:"
echo "    sage install"
echo ""
