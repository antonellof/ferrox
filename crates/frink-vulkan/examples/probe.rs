//! `cargo run -p frink-vulkan --features vulkan --example probe`
//!
//! Prints the Vulkan device this machine exposes, or names what is
//! missing. The one command to run before believing anything else in
//! this crate.
fn main() {
    match frink_vulkan::device::probe() {
        Ok(name) => println!("vulkan device: {name}"),
        Err(e) => {
            println!("no vulkan device: {e}");
            std::process::exit(1);
        }
    }
}
