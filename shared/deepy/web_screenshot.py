# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Screenshots of public web pages for Deepy's web toolbox. Playwright (Apache-2.0) drives the Edge or Chrome installed on the machine (or
Playwright's own Chromium) with a fresh profile: the user's browser profile, passwords and sign-ins are never used, only the session's cookie
jar. All the browser's traffic goes through a small proxy running beside it, which connects only to public addresses it resolved itself: every
request of the page, each redirect hop, WebSockets and HTTPS tunnels alike, so a page cannot make the browser reach WanGP's server or the local
network. The proxy relays HTTPS as it is (sites see the browser's own connection); service workers and media streams are refused."""
from __future__ import annotations

import asyncio
import importlib.util
import ipaddress
import os
import socket
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit

NAVIGATION_TIMEOUT_S = 30
SETTLE_S = 8  # waiting for the network to calm down after load; pages that keep polling are captured anyway
REQUEST_TIMEOUT_S = 20
FULL_PAGE_MAX_PX = 12000  # a full-page capture is cut there (an endless feed would never end)
_LAUNCH_ARGS = ["--force-webrtc-ip-handling-policy=disable_non_proxied_udp", "--webrtc-ip-handling-policy=disable_non_proxied_udp", "--disable-features=MediaRouter",
                "--proxy-bypass-list=<-loopback>"]  # loopback too goes through the proxy, which refuses it
_SCROLL_THROUGH = """async (limit) => {
  for (let y = 0; y < Math.min(document.documentElement.scrollHeight, limit); y += innerHeight) { scrollTo(0, y); await new Promise(done => setTimeout(done, 150)); }
  scrollTo(0, 0);
}"""
# Bars that would repeat or float in a picture joined from windows: in the first window, fixed bars in its lower half (promotions, app banners);
# in later ones every fixed bar and sticky element stuck at its offset (a header following the scroll). Sticky blocks inside the page stay.
_HIDE_BARS = """(firstWindow) => {
  for (const node of document.querySelectorAll('body *')) {
    const style = getComputedStyle(node);
    if (style.position !== 'fixed' && style.position !== 'sticky') continue;
    const top = node.getBoundingClientRect().top;
    const bar = firstWindow ? style.position === 'fixed' && top > innerHeight / 2 : style.position === 'fixed' || Math.abs(top - (parseFloat(style.top) || 0)) < 2;
    if (bar) node.style.setProperty('visibility', 'hidden', 'important');
  }
}"""
# Consent dialogs and paywall layers cover what the user wants to see: remove the known consent providers' containers and any fixed layer stacked
# above the page (z-index 100 or more) over a quarter of the window, then let the page scroll again. Sticky elements are the page's own headers and
# bars, never overlays; low layers are ad slots. Nothing is clicked, so no consent is given on the user's behalf.
_HIDE_OVERLAYS = """() => {
  const known = '#didomi-host, #onetrust-consent-sdk, #CybotCookiebotDialog, #usercentrics-root, .qc-cmp2-container, .fc-consent-root, [id^="sp_message_container"], #truste-consent-track, #cmpbox, #cmpbox2, .cc-window, #cookie-law-info-bar';
  let removed = 0;
  for (const node of document.querySelectorAll(known)) { node.remove(); removed += 1; }
  for (const node of document.querySelectorAll('body *')) {
    const style = getComputedStyle(node);
    if (style.position !== 'fixed' || (parseInt(style.zIndex) || 0) < 100) continue;
    const box = node.getBoundingClientRect();
    if (box.width * box.height > innerWidth * innerHeight * 0.25) { node.remove(); removed += 1; }
  }
  for (const node of [document.documentElement, document.body]) {  // undo a dialog's scroll lock only: forcing overflow on both would make the body its own scroller
    if (!node) continue;
    const style = getComputedStyle(node);
    if (style.overflowY === 'hidden' || style.overflow === 'hidden') node.style.setProperty('overflow', 'visible', 'important');
    if (style.position === 'fixed') node.style.setProperty('position', 'static', 'important');
  }
  return removed;
}"""


def available() -> bool:
    return importlib.util.find_spec("playwright") is not None


def system_browser() -> str | None:
    """A Chromium-family browser installed on the machine (Edge, Chrome or Chromium), so nothing has to be downloaded."""
    if sys.platform == "win32":
        bases = [os.environ.get(key) for key in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
        candidates = [Path(base) / relative for base in bases if base for relative in ("Microsoft/Edge/Application/msedge.exe", "Google/Chrome/Application/chrome.exe")]
    elif sys.platform == "darwin":
        candidates = [Path(f"/Applications/{name}.app/Contents/MacOS/{name}") for name in ("Microsoft Edge", "Google Chrome", "Chromium")]
    else:
        candidates = [Path(found) for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge") if (found := shutil.which(name))]
    return next((str(candidate) for candidate in candidates if candidate.is_file()), None)


def _browser_cookies(cookies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The session's jar in Playwright's form: host-only cookies by url, domain cookies by domain."""
    converted = []
    for cookie in cookies:
        entry: dict[str, Any] = {"name": cookie["name"], "value": cookie["value"], "secure": bool(cookie.get("secure")), "expires": float(cookie["expires"]) if cookie.get("expires") else -1}
        if cookie.get("host_only"):
            entry["url"] = f"{'https' if cookie.get('secure') else 'http'}://{cookie['domain']}{cookie.get('path') or '/'}"
        else:
            entry.update(domain="." + cookie["domain"], path=cookie.get("path") or "/")
        converted.append(entry)
    return converted


def capture(url: str, target: Path, check_public: Callable[[str], None], *, width: int, height: int, full_page: bool, wait_s: float, cookies: list[dict[str, Any]], hide_overlays: bool = True) -> dict[str, Any]:
    """Screenshot url into target (PNG), on a thread with an event loop of its own: WanGP sets the selector event loop policy on Windows,
    whose loops cannot start the browser driver's process, so this loop is a Proactor loop there (the process-wide policy is left as it is)."""
    outcome: dict[str, Any] = {}

    def work() -> None:
        loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        asyncio.set_event_loop(loop)  # this thread's own
        try:
            outcome["result"] = loop.run_until_complete(_capture(url, target, check_public, width=width, height=height, full_page=full_page, wait_s=wait_s, cookies=cookies, hide_overlays=hide_overlays))
        except BaseException as error:  # handed back to the caller's thread
            outcome["error"] = error
        finally:
            pending = asyncio.all_tasks(loop)  # proxy connections the closed browser left open
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    worker = threading.Thread(target=work, name="deepy-web-screenshot", daemon=True)
    worker.start()
    worker.join()
    error = outcome.get("error")
    if isinstance(error, ValueError):
        raise error
    if error is not None:  # some errors carry no message: the type says what failed
        raise RuntimeError(f"The browser could not take the screenshot: {type(error).__name__}: {error}".rstrip(": ")) from error
    return outcome["result"]


async def _filtering_proxy(check_public: Callable[[str], None], blocked: list[str]) -> asyncio.AbstractServer:
    """An HTTP proxy on 127.0.0.1 for the browser: CONNECT tunnels and plain HTTP requests reach a host only through an address checked as public
    (the name is resolved here and that address is the one connected to, so a page cannot slip a private address in between)."""
    loop = asyncio.get_running_loop()

    async def connect(host: str, port: int):
        try:
            await loop.run_in_executor(None, check_public, f"https://{host}:{port}/")
            infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except (ValueError, OSError):
            infos = []
        for _family, _type, _proto, _name, address in infos:
            if ipaddress.ip_address(address[0].split("%", 1)[0]).is_global:
                return await asyncio.wait_for(asyncio.open_connection(address[0], address[1]), REQUEST_TIMEOUT_S)
        if host not in blocked and len(blocked) < 5:
            blocked.append(host)
        return None

    async def pipe(source, destination) -> None:
        try:
            while data := await source.read(1 << 16):
                destination.write(data)
                await destination.drain()
        except (OSError, asyncio.IncompleteReadError):
            pass
        finally:
            destination.close()

    async def handle(reader, writer) -> None:
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1")
            method, target, version = head.split("\r\n", 1)[0].split(" ", 2)
            if method == "CONNECT":  # HTTPS and secure WebSockets: the tunnel carries the browser's own encrypted connection
                host, _, port = target.rpartition(":")
                upstream = await connect(host.strip("[]"), int(port))
                if upstream is None:
                    writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                    writer.close()
                    return
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            else:  # plain HTTP, one request per connection so each one is checked
                parts = urlsplit(target)
                upstream = await connect(parts.hostname or "", parts.port or 80)
                if upstream is None:
                    writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    writer.close()
                    return
                lines = [line for line in head.split("\r\n")[1:] if line and line.split(":", 1)[0].strip().lower() not in ("proxy-connection", "connection", "keep-alive")]
                request = f"{method} {(parts.path or '/') + ('?' + parts.query if parts.query else '')} {version}\r\n" + "".join(line + "\r\n" for line in lines) + "Connection: close\r\n\r\n"
                upstream[1].write(request.encode("latin-1"))
            await asyncio.gather(pipe(reader, upstream[1]), pipe(upstream[0], writer))
        except (OSError, ValueError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError):
            writer.close()

    return await asyncio.start_server(handle, "127.0.0.1", 0)


async def _capture_tiles(page, target: Path, width: int, height: int) -> None:
    """The top FULL_PAGE_MAX_PX of a very tall page, window by window: the browser cannot paint a document of tens of thousands of pixels at once
    (a clipped full-page capture comes back gray below the first screen). The page's header bars stay in the first window only; bars fixed to the
    bottom of the window (promotions, app banners) are left out, or they would sit in the middle of the picture."""
    from io import BytesIO

    from PIL import Image

    canvas = Image.new("RGB", (width, FULL_PAGE_MAX_PX), "white")
    await page.evaluate(_HIDE_BARS, True)
    for top in range(0, FULL_PAGE_MAX_PX, height):
        await page.evaluate("top => scrollTo({top, behavior: 'instant'})", top)
        await page.wait_for_timeout(200)
        if top:  # bars some pages only show once scrolled are hidden too
            await page.evaluate(_HIDE_BARS, False)
        with Image.open(BytesIO(await page.screenshot(type="png"))) as window:
            canvas.paste(window.convert("RGB"), (0, int(await page.evaluate("scrollY"))))
    canvas.save(target)


async def _capture(url: str, target: Path, check_public: Callable[[str], None], *, width: int, height: int, full_page: bool, wait_s: float, cookies: list[dict[str, Any]], hide_overlays: bool) -> dict[str, Any]:
    from playwright.async_api import Error as PlaywrightError, TimeoutError as PlaywrightTimeout, async_playwright

    blocked: list[str] = []
    proxy = await _filtering_proxy(check_public, blocked)
    proxy_port = proxy.sockets[0].getsockname()[1]

    async def route(route_, request) -> None:  # media streams are not needed for a still and would be read whole
        await (route_.abort("blockedbyclient") if request.resource_type == "media" else route_.continue_())

    try:
        async with async_playwright() as playwright:
            executable = system_browser()
            if executable is None and not Path(playwright.chromium.executable_path).is_file():
                raise ValueError("Web screenshots need a Chromium-family browser: install Microsoft Edge or Google Chrome, or run `python -m playwright install chromium` once.")
            browser = await playwright.chromium.launch(executable_path=executable, headless=True, args=[*_LAUNCH_ARGS, f"--proxy-server=http://127.0.0.1:{proxy_port}"])
            try:
                context = await browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=1, accept_downloads=False, service_workers="block")
                if cookies:
                    await context.add_cookies(_browser_cookies(cookies))
                await context.route("**/*", route)
                page = await context.new_page()
                try:
                    response = await page.goto(url, wait_until="load", timeout=NAVIGATION_TIMEOUT_S * 1000)
                except PlaywrightError as error:
                    reason = str(error).splitlines()[0].removeprefix("Page.goto: ")
                    raise ValueError(f"The page could not be loaded ({reason})" + (f": it leads to a local or private address ({', '.join(blocked)}), which cannot be opened." if blocked else ".")) from None
                try:
                    await page.wait_for_load_state("networkidle", timeout=SETTLE_S * 1000)
                except PlaywrightTimeout:  # a page that keeps polling never goes idle: it is captured as it is
                    pass
                if full_page:  # images that load when scrolled into view: go down the page once so they are drawn, then back to the top
                    await page.evaluate(_SCROLL_THROUGH, FULL_PAGE_MAX_PX)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=4000)
                    except PlaywrightTimeout:
                        pass
                    await page.wait_for_timeout(800)  # headers that slid away while scrolling down come back
                if wait_s > 0:
                    await page.wait_for_timeout(wait_s * 1000)
                overlays = int(await page.evaluate(_HIDE_OVERLAYS)) if hide_overlays else 0
                page_height = int(await page.evaluate("Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0)") or height)
                cut = full_page and page_height > FULL_PAGE_MAX_PX
                target.parent.mkdir(parents=True, exist_ok=True)
                if cut:
                    await _capture_tiles(page, target, width, height)
                else:
                    await page.screenshot(path=str(target), type="png", full_page=full_page)
                result = {"final_url": page.url, "title": await page.title(), "http_status": response.status if response is not None else None, "page_height": page_height, "cut": cut, "overlays_removed": overlays, "blocked_hosts": blocked}
            finally:
                await browser.close()
    finally:
        proxy.close()
    result["browser"] = Path(executable).stem if executable else "chromium"
    return result
