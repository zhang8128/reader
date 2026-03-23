# Reader

Epub reader for shared reading sessions with engram memory.

## What it does

Parses epubs into semantically-chunked passages and serves them one at a time.
Reactions are stored locally and forwarded to engram as memories and dream captures.

## Reading flow

```
open_book("mistborn-1.epub")  → parse, chunk, store
read()                         → next passage
[react in conversation]
react("...")                   → store reaction, get engram forwarding instructions
[call engram.remember() and engram.dream_capture() with the returned content]
read()                         → next passage
```

## Files

| File | Role |
|------|------|
| `epub.py` | Epub parsing — zip, spine order, HTML stripping |
| `chunker.py` | Semantic chunking via embedding similarity drops |
| `store.py` | SQLite reading state — books, chunks, position, reactions |
| `server.py` | MCP server — open_book, read, react |

## Dependencies

fastembed (nomic-embed-text-v1.5), beautifulsoup4, lxml, mcp
