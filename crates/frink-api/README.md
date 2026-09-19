# frink-api

The wire contract between `frink-server` and its clients.

Route constants and serde DTOs shared by the server, Frink Studio (the
separate web app in `ui/`), the desktop shell that spawns the server,
and `frink chat`. A path or a payload shape gets exactly one
definition instead of one per end.

Contents:

- `routes`: every path the server serves, named once.
- `health`: the three-state `/health` handshake (`ready`, `detecting`,
  `unavailable`) with a machine reason code and a human sentence per
  capability.
- `lifecycle`: the `frink.server.ready` stdout line carrying the
  address the server ended up bound to and its pid, which is what makes
  `--port 0` usable by a supervising process.
- `usage`: token accounting plus llama.cpp-style timings, with prefill
  and decode kept separate.
- `request_id`: server-assigned ids, stated in the first streamed chunk.
- `progress`: a rolling-window transfer rate and ETA that reports
  nothing until it has something true to report.

Depends on serde and nothing else, so a CLI, a desktop shell and a
browser-targeted frontend all link it.
