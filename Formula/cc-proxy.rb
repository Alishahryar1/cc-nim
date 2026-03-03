class CcProxy < Formula
  include Language::Python::Virtualenv

  desc "Claude Code proxy for NVIDIA NIM, OpenRouter, and LM Studio"
  homepage "https://github.com/Alishahryar1/free-claude-code"
  url "https://github.com/Alishahryar1/free-claude-code/archive/b49e91a.tar.gz"
  sha256 "fce7cebffdb352e2d4c4b9eab3faab174aa924aaf5a6ff4c514ae15f60ef37bb"
  license "MIT"
  head "https://github.com/Alishahryar1/free-claude-code.git", branch: "main"

  depends_on "python@3.14"

  def install
    virtualenv_install_with_resources
  end

  def post_install
    config = Pathname.new(Dir.home) / ".ccenv"
    system bin/"cc-nim", "init" unless config.exist?
  end

  service do
    run [opt_bin/"cc-nim", "start"]
    keep_alive true
    log_path var/"log/cc-proxy.log"
    error_log_path var/"log/cc-proxy-error.log"
  end

  test do
    output = shell_output("#{bin}/cc-nim 2>&1", 1)
    assert_match "Usage: cc-nim", output
  end
end
