//! The CLI's minimal HTTP/1.1 client, shared by `frink chat` and
//! `frink serve-bench`.
//!
//! Hand-rolled on `TcpStream` on purpose: the CLI links no HTTP stack,
//! and pulling one in for two commands that talk to one local server
//! over plain `http://` would add a TLS stack, an async runtime and
//! their dependency trees to every `frink` build.
//!
//! It exists as its own module because both callers had grown their own
//! copy of the same twenty lines -- `parse_url` was duplicated
//! character for character, and the request preamble three times. Two
//! copies of a client is how one of them keeps a timeout the other
//! drops, which then shows up as a benchmark that hangs where the chat
//! client would have given up.
//!
//! # What it does not do
//!
//! No TLS, no redirects, no keep-alive (every request sends
//! `Connection: close`), no chunked-transfer decoding. The last one is
//! safe only because the callers read either a whole body to EOF or an
//! SSE stream line by line, and both are correct under `close`
//! framing -- a caller that needs to parse a chunked response would
//! have to add it.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};

/// Long enough for a cold prefill on a large model to produce its first
/// token. A benchmark that gives up early reports a failure the server
/// did not have.
const READ_TIMEOUT: Duration = Duration::from_secs(600);

/// The request body is small and local; a write that blocks this long
/// is a dead peer, not a slow one.
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);

/// A parsed `http://host[:port][/path]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Url {
    pub host: String,
    pub port: u16,
    pub path: String,
}

/// A buffered response: status plus the whole body.
pub struct Response {
    pub status: u16,
    pub body: Vec<u8>,
}

impl Response {
    /// The body as an error when the status is not 2xx, so a caller can
    /// fail with what the server actually said rather than with a bare
    /// status code.
    pub fn ok_or_status(self) -> Result<Vec<u8>> {
        if (200..300).contains(&self.status) {
            return Ok(self.body);
        }
        let msg = String::from_utf8_lossy(&self.body);
        anyhow::bail!("HTTP {}: {msg}", self.status)
    }
}

pub fn parse_url(url: &str) -> Result<Url> {
    let rest = url.strip_prefix("http://").ok_or_else(|| {
        anyhow!("only http:// URLs are supported (got {url}); https is not implemented in the CLI")
    })?;
    let (hostport, path) = match rest.split_once('/') {
        Some((hp, p)) => (hp, format!("/{p}")),
        None => (rest, "/".to_string()),
    };
    let (host, port) = match hostport.split_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().context("bad port")?),
        None => (hostport.to_string(), 80u16),
    };
    Ok(Url { host, port, path })
}

/// Sends one request and returns the status plus a reader positioned at
/// the first byte of the body.
///
/// This is the streaming entry point: an SSE caller reads lines from
/// the reader as they arrive, which is the whole reason the response is
/// handed back unread rather than buffered.
pub fn open(method: &str, url: &str, body: Option<&[u8]>) -> Result<(u16, BufReader<TcpStream>)> {
    let Url { host, port, path } = parse_url(url)?;
    let mut stream = TcpStream::connect((host.as_str(), port))
        .with_context(|| format!("connect {host}:{port}"))?;
    stream.set_read_timeout(Some(READ_TIMEOUT))?;
    stream.set_write_timeout(Some(WRITE_TIMEOUT))?;

    let mut req =
        format!("{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n");
    match body {
        Some(b) => {
            req.push_str("Content-Type: application/json\r\n");
            req.push_str(&format!("Content-Length: {}\r\n\r\n", b.len()));
            stream.write_all(req.as_bytes())?;
            stream.write_all(b)?;
        }
        None => {
            req.push_str("\r\n");
            stream.write_all(req.as_bytes())?;
        }
    }

    let mut reader = BufReader::new(stream);
    let status = read_status_and_headers(&mut reader)?;
    Ok((status, reader))
}

/// Sends one request and reads the whole body.
pub fn exchange(method: &str, url: &str, body: Option<&[u8]>) -> Result<Response> {
    let (status, mut reader) = open(method, url, body)?;
    let mut body = Vec::new();
    reader.read_to_end(&mut body)?;
    Ok(Response { status, body })
}

pub fn get(url: &str) -> Result<Response> {
    exchange("GET", url, None)
}

/// Reads the status line and discards headers, leaving the reader at
/// the body.
///
/// A truncated response (EOF mid-headers) ends the loop rather than
/// spinning: `read_line` returns `Ok(0)` forever at EOF, so a loop that
/// only breaks on a blank line would never return.
fn read_status_and_headers(reader: &mut BufReader<TcpStream>) -> Result<u16> {
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
    }
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three shapes a `--url` actually arrives in, and the default
    /// port. A missing port defaulting to anything but 80 would make
    /// `http://host/path` reach a different server than the same URL in
    /// a browser.
    #[test]
    fn a_url_splits_into_host_port_and_path() {
        assert_eq!(
            parse_url("http://127.0.0.1:8383").unwrap(),
            Url {
                host: "127.0.0.1".into(),
                port: 8383,
                path: "/".into()
            }
        );
        assert_eq!(
            parse_url("http://127.0.0.1:8383/v1/models").unwrap(),
            Url {
                host: "127.0.0.1".into(),
                port: 8383,
                path: "/v1/models".into()
            }
        );
        assert_eq!(
            parse_url("http://example").unwrap(),
            Url {
                host: "example".into(),
                port: 80,
                path: "/".into()
            }
        );
    }

    /// https is refused by name rather than attempted and failed at the
    /// socket, where the error would be a TLS handshake mystery.
    #[test]
    fn a_non_http_url_is_refused_by_name() {
        let err = parse_url("https://example.com").unwrap_err().to_string();
        assert!(err.contains("https"), "{err}");
        assert!(parse_url("127.0.0.1:8383").is_err(), "no scheme at all");
    }

    /// A bad port is an error, not a silent fallback to 80: a caller
    /// that typed the port meant it, and quietly reaching a different
    /// server is worse than failing.
    #[test]
    fn an_unparseable_port_is_an_error() {
        assert!(parse_url("http://host:not-a-port/").is_err());
    }

    /// A non-2xx response carries the server's own message, which for
    /// this server is a JSON error naming what was wrong.
    #[test]
    fn a_failed_response_reports_what_the_server_said() {
        let err = Response {
            status: 400,
            body: b"{\"error\":\"max_tokens must be positive\"}".to_vec(),
        }
        .ok_or_status()
        .unwrap_err()
        .to_string();
        assert!(err.contains("400"), "{err}");
        assert!(err.contains("max_tokens"), "{err}");

        let ok = Response {
            status: 200,
            body: b"{}".to_vec(),
        }
        .ok_or_status()
        .unwrap();
        assert_eq!(ok, b"{}");
    }
}
