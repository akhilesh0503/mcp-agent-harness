import json
import ipaddress
import urllib.parse
import httpx

_BLOCKED_HOSTS = {"localhost", "0.0.0.0"}
_ALLOWED_METHODS = {"GET", "POST"}
_TIMEOUT = 30.0


def _is_internal_ip(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        return False


def _validate_url(url: str) -> str | None:
    """Return an error string if the URL should be blocked, else None."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return "Invalid URL"

    if parsed.scheme not in ("http", "https"):
        return "Only http and https schemes are allowed"

    host = parsed.hostname or ""
    if host in _BLOCKED_HOSTS or _is_internal_ip(host):
        return f"Requests to internal/private hosts are not allowed: {host}"

    return None


async def http_api_call_tool(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    body: dict | None = None,
) -> str:
    method = method.upper()
    if method not in _ALLOWED_METHODS:
        return json.dumps({"error": f"Method not allowed. Use one of: {sorted(_ALLOWED_METHODS)}"})

    err = _validate_url(url)
    if err:
        return json.dumps({"error": err})

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers or {},
                json=body if method == "POST" and body else None,
            )
        try:
            body_content = response.json()
        except Exception:
            body_content = response.text

        return json.dumps({
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": body_content,
        }, default=str)

    except httpx.TimeoutException:
        return json.dumps({"error": f"Request timed out after {_TIMEOUT}s"})
    except httpx.RequestError as e:
        return json.dumps({"error": str(e)})
