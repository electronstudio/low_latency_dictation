# Maintainer: TODO <your-email>
pkgname=low-latency-dictation-git
pkgver=0  # placeholder, overwritten by pkgver()
pkgrel=1
pkgdesc="Real-time microphone speech-to-text using whisper.cpp (local inference)"
arch=('x86_64')
url="https://github.com/electronstudio/low_latency_dictation"
license=('GPL-3.0-only')
depends=('sdl2' 'vulkan-icd-loader' 'libnotify' 'gcc-libs')
makedepends=('go>=1.24' 'cmake' 'git' 'vulkan-headers' 'shaderc')
install="$pkgname.install"
source=(
  "git+https://github.com/electronstudio/low_latency_dictation.git"
)
sha256sums=('SKIP')

pkgver() {
  cd "$srcdir/low_latency_dictation"
  local base=$(cat VERSION)
  printf "%s.r%s.%s" "$base" "$(git rev-list --count HEAD)" "$(git rev-parse --short HEAD)"
}

prepare() {
  cd "$srcdir/low_latency_dictation"
  git submodule update --init --recursive
}

build() {
  cd "$srcdir/low_latency_dictation"
  export GOFLAGS=-buildvcs=false
  make -f Makefile.linux
}

package() {
  cd "$srcdir/low_latency_dictation"
  make -f Makefile.linux install DESTDIR="$pkgdir" PREFIX=/usr
  rm -f "$pkgdir/usr/share/applications/mimeinfo.cache"
  rm -f "$pkgdir/usr/share/icons/hicolor/icon-theme.cache"
}
