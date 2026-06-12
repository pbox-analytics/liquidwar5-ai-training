# Liquid War — mobile shells (Capacitor)

Thin native wrappers for iOS and Android: the app is a WebView shell that
loads the live game from the LAN server (`capacitor.config.json` →
`server.url`). Game updates deploy server-side as always — the apps never
need rebuilding for gameplay changes.

## Android

Built in CI-style via the dockerized SDK (no local toolchain):

    docker run --rm -v "$PWD/android:/project" -w /project \
      cimg/android:2024.10 ./gradlew --no-daemon assembleDebug

APK lands at `android/app/build/outputs/apk/debug/app-debug.apk`.
The deploy flow copies it to the play server as `/liquidwar.apk`, so a phone
can download it straight from the game page (enable "install unknown apps").

## iOS (needs a Mac with Xcode)

    cd mobile && npx cap sync ios && npx cap open ios

Then run on a device from Xcode (free Apple ID = 7-day signing; a $99
developer account removes the expiry). `NSAppTransportSecurity` exceptions
for the cleartext LAN server are handled by Capacitor's `cleartext` config.

## Pointing at a different server

Edit `server.url` in `capacitor.config.json`, then `npx cap sync`.
