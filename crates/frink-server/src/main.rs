//! Everything the server is lives in `lib.rs`; this binary exists so
//! `cargo install frink-server` keeps producing a `frink-server`
//! executable. The same library backs frink-cli's optional `serve`
//! feature, which is why there is no logic here to drift.

fn main() -> anyhow::Result<()> {
    frink_server::run_server(frink_server::ServerArgs::parse_llama_style(std::env::args()))
}
