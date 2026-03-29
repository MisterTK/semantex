# This formula lives in the homebrew-semantex tap repo:
#   https://github.com/MisterTK/homebrew-semantex
#
# Install:
#   brew tap MisterTK/semantex
#   brew install semantex

class Semantex < Formula
  desc "Semantic code search — hybrid ColBERT + BM25 retrieval engine"
  homepage "https://semantex.dev"
  version "0.1.2"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/MisterTK/semantex/releases/download/v#{version}/semantex-v#{version}-aarch64-apple-darwin.tar.gz"
      sha256 "AARCH64_APPLE_DARWIN_SHA256"
    end
    on_intel do
      url "https://github.com/MisterTK/semantex/releases/download/v#{version}/semantex-v#{version}-x86_64-apple-darwin.tar.gz"
      sha256 "X86_64_APPLE_DARWIN_SHA256"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/MisterTK/semantex/releases/download/v#{version}/semantex-v#{version}-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "AARCH64_LINUX_SHA256"
    end
    on_intel do
      url "https://github.com/MisterTK/semantex/releases/download/v#{version}/semantex-v#{version}-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "X86_64_LINUX_SHA256"
    end
  end

  def install
    bin.install "semantex"
    # ONNX Runtime dylib must live next to the binary for runtime loading
    Dir["libonnxruntime*"].each { |lib| bin.install lib }
  end

  def caveats
    <<~EOS
      To integrate with your AI coding tool, run:
        semantex install-claude-code   # Claude Code
        semantex install-codex         # Codex CLI
        semantex install-opencode      # OpenCode

      To disable anonymous usage telemetry:
        export SEMANTEX_NO_TELEMETRY=1
    EOS
  end

  test do
    system "#{bin}/semantex", "--version"
  end
end
