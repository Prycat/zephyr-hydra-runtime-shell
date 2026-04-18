# Zephyr Tool Expansion — Design Doc
Date: 2026-04-10
Status: Approved

## Goal
Add 4 new tools to Zephyr (agent.py) so he can search the web, read URLs,
run Python code, and make HTTP API calls — all with zero API keys required.

## Tools

### 1. web_search(query, max_results=5)
- Library: `duckduckgo-search` (free, no key)
- Returns: top N results as title + url + snippet
- Timeout: 10s

### 2. browse_url(url)
- Libraries: `httpx` (already installed) + `beautifulsoup4`
- Strips HTML tags, returns clean readable text
- Truncates to 3000 chars to stay within context window
- Timeout: 10s

### 3. run_python(code)
- Uses `subprocess` to run code in an isolated Python process via temp file
- Captures stdout + stderr
- Hard timeout: 10s (prevents infinite loops)
- Returns combined output or error message

### 4. http_request(method, url, headers, body)
- Uses `httpx` (already installed)
- Supports GET/POST/PUT/DELETE
- Returns: status code + response body (pretty-printed JSON if parseable)
- Timeout: 15s

## Dependencies
- `duckduckgo-search` — new install required
- `beautifulsoup4` — new install required
- `httpx`, `subprocess`, `tempfile` — already available

## Testing Plan
Run a dedicated test script (test_tools.py) that calls each tool function
directly and prints pass/fail. No LLM needed for tool unit tests.
