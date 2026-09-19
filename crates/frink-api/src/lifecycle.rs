//! The process-ready handshake: one machine-readable line on the
//! server's stdout, naming the address it actually bound and the pid
//! that owns it.
//!
//! This is what makes `--port 0` usable, and `--port 0` is what deletes
//! an entire class of feature from the desktop shell. A supervisor that
//! must pick the port itself needs to know whether the port is free,
//! which needs an "is something already listening" probe, which needs
//! "who owns it" to tell a stale copy of ourselves from a stranger's
//! server, which needs a platform-specific `lsof`/`netstat` shell-out
//! and a dialog to explain the result. Letting the kernel pick the port
//! and having the child *say* what it got replaces all of it with one
//! line of JSON.
//!
//! One line, JSON, on stdout, prefixed by nothing: parsers should read
//! stdout line by line and ignore any line that is not JSON carrying
//! `event == `[`READY_EVENT`], since tracing/log output shares the
//! stream on some configurations.

use serde::{Deserialize, Serialize};

/// The `event` discriminator of the ready line. Present so a parser can
/// tell this line from any other JSON a future version might print.
pub const READY_EVENT: &str = "frink.server.ready";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerReady {
    /// Always [`READY_EVENT`].
    pub event: String,
    /// `host:port` as actually bound -- never the requested value, which
    /// may have been port 0.
    pub addr: String,
    /// The bound port, split out so a caller does not have to parse
    /// `addr` (IPv6 literals make that its own small mistake).
    pub port: u16,
    /// `http` or `https`, so a client can build a base URL without
    /// guessing whether TLS was configured.
    pub scheme: String,
    pub pid: u32,
    pub version: String,
}

impl ServerReady {
    pub fn new(addr: std::net::SocketAddr, scheme: &str, version: &str, pid: u32) -> Self {
        ServerReady {
            event: READY_EVENT.to_string(),
            addr: addr.to_string(),
            port: addr.port(),
            scheme: scheme.to_string(),
            pid,
            version: version.to_string(),
        }
    }

    /// The exact bytes to print (no trailing newline).
    pub fn to_line(&self) -> String {
        serde_json::to_string(self).expect("ServerReady is plain data and cannot fail to serialize")
    }

    /// Parses one line of a child's stdout. `None` for anything that is
    /// not a ready line, so a caller can feed it every line it reads.
    pub fn from_line(line: &str) -> Option<Self> {
        let parsed: ServerReady = serde_json::from_str(line.trim()).ok()?;
        (parsed.event == READY_EVENT).then_some(parsed)
    }

    /// Base URL for API calls against this server.
    pub fn base_url(&self) -> String {
        format!("{}://{}", self.scheme, self.addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

    fn ready(port: u16) -> ServerReady {
        ServerReady::new(
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port),
            "http",
            "0.5.0",
            4242,
        )
    }

    #[test]
    fn round_trips_through_one_stdout_line() {
        let line = ready(51234).to_line();
        assert!(!line.contains('\n'), "the ready line must be a single line");
        assert_eq!(ServerReady::from_line(&line), Some(ready(51234)));
    }

    #[test]
    fn ignores_lines_that_are_not_the_ready_event() {
        assert!(ServerReady::from_line("2026-08-14 INFO listening").is_none());
        assert!(ServerReady::from_line("{\"event\":\"something.else\"}").is_none());
        assert!(ServerReady::from_line("").is_none());
    }

    #[test]
    fn port_survives_an_ipv6_address_without_parsing_addr() {
        // The reason `port` is its own field: splitting an IPv6 `addr`
        // on ':' finds the wrong colon.
        let ready = ServerReady::new(
            SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 8383),
            "https",
            "0.5.0",
            7,
        );
        assert_eq!(ready.port, 8383);
        assert_eq!(ready.base_url(), "https://[::1]:8383");
    }
}
