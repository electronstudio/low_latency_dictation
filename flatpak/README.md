# Flatpak

May not be suitable for submission to Flathub yet.

## Build and install

    rm -rf .flatpak-builder repo builddir build-dir
    flatpak-builder --install --install-deps-from=flathub --force-clean --user builddir uk.co.electronstudio.Dictate.yaml

## Run

    flatpak run uk.co.electronstudio.Dictate

## Validation checks

    flatpak run --command=flatpak-builder-lint org.flatpak.Builder manifest uk.co.electronstudio.Dictate.yaml
    flatpak run --command=flatpak-builder-lint org.flatpak.Builder appstream uk.co.electronstudio.Dictate.metainfo.xml
    flatpak run --command=flatpak-builder-lint org.flatpak.Builder builddir builddir
    flatpak run --command=flatpak-builder-lint org.flatpak.Builder repo repo

## Standalone bundle

    flatpak build-bundle repo low_latency_dictation.flatpak uk.co.electronstudio.Dictate
