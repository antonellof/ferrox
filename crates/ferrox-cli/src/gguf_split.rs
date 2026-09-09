//! `ferrox gguf-split`: llama.cpp's `llama-gguf-split`, same flags,
//! same shard names, same `split.*` keys. The work is in
//! `ferrox_gguf::split`; this module is argument parsing and the
//! progress lines.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use ferrox_gguf::split::{
    plan_merge, plan_split, write_merge, write_split, SplitMode, SplitOptions, SplitPlan,
    DEFAULT_MAX_TENSORS,
};
use ferrox_gguf::GgufFile;

#[derive(Parser, Debug)]
pub struct GgufSplitArgs {
    /// Split GGUF_IN into `GGUF_OUT-NNNNN-of-MMMMM.gguf` shards. The
    /// default when neither operation is given.
    #[arg(long, conflicts_with = "merge")]
    pub split: bool,

    /// Merge the shard set whose FIRST shard is GGUF_IN into GGUF_OUT.
    #[arg(long)]
    pub merge: bool,

    /// Max tensors per shard (llama.cpp's default: 128).
    ///
    /// `Option`, not a `default_value_t`: `--merge` refuses every split
    /// option it was given, and a defaulted field cannot say whether
    /// the user typed it.
    #[arg(long, conflicts_with = "split_max_size")]
    pub split_max_tensors: Option<usize>,

    /// Max tensor bytes per shard, as `N(M|G)`: decimal megabytes or
    /// gigabytes, the way llama.cpp reads them (`128M` is 128,000,000).
    #[arg(long, value_parser = parse_split_size)]
    pub split_max_size: Option<u64>,

    /// Leave the first shard metadata-only, the layout most published
    /// multi-file checkpoints use.
    #[arg(long)]
    pub no_tensor_first_split: bool,

    /// Print the split plan (shard count, tensors and size per shard)
    /// and write nothing.
    #[arg(long)]
    pub dry_run: bool,

    /// Source GGUF for `--split`; first shard (`...-00001-of-MMMMM.gguf`)
    /// for `--merge`.
    pub input: PathBuf,

    /// Output prefix for `--split` (`out/model` writes
    /// `out/model-00001-of-00003.gguf`, ...); output file for `--merge`.
    pub output: PathBuf,
}

/// `split_str_to_n_bytes`, `gguf-split.cpp:72-88`: a positive integer
/// followed by `M` (10^6) or `G` (10^9). Lower case is accepted too.
pub fn parse_split_size(s: &str) -> std::result::Result<u64, String> {
    let (digits, unit) = match s.char_indices().last() {
        Some((i, c)) => (&s[..i], c.to_ascii_uppercase()),
        None => return Err("expected N(M|G), for example 4G".into()),
    };
    let multiplier: u64 = match unit {
        'M' => 1_000_000,
        'G' => 1_000_000_000,
        other => {
            return Err(format!(
                "supported units are M (megabytes) or G (gigabytes), got '{other}'"
            ))
        }
    };
    let n: u64 = digits
        .parse()
        .map_err(|_| format!("'{digits}' is not a whole number"))?;
    if n == 0 {
        return Err("size must be a positive value".into());
    }
    n.checked_mul(multiplier)
        .ok_or_else(|| format!("{s} does not fit in 64 bits"))
}

impl GgufSplitArgs {
    /// The one place the two size flags become a mode:
    /// `--split-max-size` wins when given, otherwise tensor mode, and
    /// tensor mode is the default with llama.cpp's 128
    /// (`gguf-split.cpp:45`, `:161-164`).
    pub fn mode(&self) -> SplitMode {
        match self.split_max_size {
            Some(bytes) => SplitMode::MaxBytes(bytes),
            None => SplitMode::MaxTensors(self.split_max_tensors.unwrap_or(DEFAULT_MAX_TENSORS)),
        }
    }

    /// Every option that only means something to `--split`, named,
    /// with whether it was typed. `--merge` refuses the ones it was
    /// given from THIS list, and the tests walk it, so a flag cannot be
    /// added to one and forgotten by the other.
    ///
    /// The destructure is exhaustive with no `..` on purpose: a new
    /// field on `GgufSplitArgs` stops compiling here until someone
    /// decides which half of the tool it belongs to. The alternative
    /// ships a split-only flag that `--merge` accepts and ignores,
    /// which is this repo's dominant bug shape.
    fn split_only_flags(&self) -> [(&'static str, bool); 3] {
        let GgufSplitArgs {
            split: _,
            merge: _,
            split_max_tensors,
            split_max_size,
            no_tensor_first_split,
            dry_run: _,
            input: _,
            output: _,
        } = self;
        [
            ("--split-max-tensors", split_max_tensors.is_some()),
            ("--split-max-size", split_max_size.is_some()),
            ("--no-tensor-first-split", *no_tensor_first_split),
        ]
    }
}

pub fn run(args: GgufSplitArgs) -> Result<()> {
    if args.merge {
        run_merge(&args)
    } else {
        run_split(&args)
    }
}

/// llama.cpp's `print_info` (`gguf-split.cpp:294-308`), plus the file
/// size, since the tool's reason to exist is fitting a size limit.
fn print_plan(plan: &SplitPlan) {
    println!("n_split: {}", plan.shards.len());
    for (i, shard) in plan.shards.iter().enumerate() {
        println!(
            "split {:05}: n_tensors = {}, total_size = {}M (file {} bytes)",
            i + 1,
            shard.tensors.len(),
            (shard.header_bytes as u64 + shard.data_bytes) / 1_000_000,
            shard.file_bytes
        );
    }
}

fn run_split(args: &GgufSplitArgs) -> Result<()> {
    let source =
        GgufFile::open(&args.input).with_context(|| format!("opening {}", args.input.display()))?;
    let opts = SplitOptions {
        mode: args.mode(),
        no_tensor_first_split: args.no_tensor_first_split,
    };
    let plan = plan_split(&source, &opts)
        .with_context(|| format!("planning a split of {}", args.input.display()))?;
    print_plan(&plan);
    if args.dry_run {
        println!("dry run: nothing written");
        return Ok(());
    }
    let paths = write_split(&source, &plan, &args.output, |path| {
        println!("Writing file {} ...", path.display());
    })?;
    eprintln!(
        "gguf_split: {} gguf split written with a total of {} tensors.",
        paths.len(),
        plan.n_tensors
    );
    Ok(())
}

fn run_merge(args: &GgufSplitArgs) -> Result<()> {
    // llama.cpp parses the split options in merge mode and then never
    // reads them, so `--merge --split-max-size 1G` silently does
    // nothing there. Refusing names what was ignored instead.
    let given: Vec<&str> = args
        .split_only_flags()
        .iter()
        .filter(|(_, typed)| *typed)
        .map(|(name, _)| *name)
        .collect();
    if !given.is_empty() {
        bail!(
            "--merge takes no split options, but {} was given; a merge writes one file, so \
             there is no shard size to choose",
            given.join(", ")
        );
    }
    eprintln!(
        "gguf_merge: {} -> {}",
        args.input.display(),
        args.output.display()
    );
    let plan = plan_merge(&args.input)?;
    for path in plan.shard_paths() {
        eprintln!("gguf_merge: reading metadata {} ... done", path.display());
    }
    println!(
        "n_split: {}, n_tensors: {}",
        plan.shard_paths().len(),
        plan.tensors.len()
    );
    if args.dry_run {
        println!("dry run: nothing written");
        return Ok(());
    }
    write_merge(&plan, &args.output)?;
    eprintln!(
        "gguf_merge: {} merged from {} split with {} tensors.",
        args.output.display(),
        plan.shard_paths().len(),
        plan.tensors.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> GgufSplitArgs {
        let mut full = vec!["gguf-split"];
        full.extend_from_slice(argv);
        GgufSplitArgs::try_parse_from(full)
            .unwrap_or_else(|e| panic!("`{}` did not parse: {e}", argv.join(" ")))
    }

    /// llama.cpp's units are decimal (`gguf-split.cpp:77`, `:80`):
    /// `4G` is 4,000,000,000, not 2^32. A binary reading would produce
    /// shards 7% larger than the limit the user typed.
    #[test]
    fn split_size_units_are_decimal_like_llama_cpp() {
        assert_eq!(parse_split_size("128M").unwrap(), 128_000_000);
        assert_eq!(parse_split_size("4G").unwrap(), 4_000_000_000);
        assert_eq!(parse_split_size("4g").unwrap(), 4_000_000_000);
        for bad in ["4", "4K", "0G", "-1G", "G", "", "4.5G"] {
            assert!(parse_split_size(bad).is_err(), "{bad:?} parsed");
        }
    }

    /// Without either flag the mode is tensor mode at llama.cpp's 128
    /// (`gguf-split.cpp:45`, `:161-164`); a size flag switches mode.
    #[test]
    fn the_default_mode_is_128_tensors_and_a_size_flag_switches_it() {
        let args = parse(&["in.gguf", "out"]);
        assert_eq!(args.mode(), SplitMode::MaxTensors(DEFAULT_MAX_TENSORS));
        assert_eq!(DEFAULT_MAX_TENSORS, 128);
        assert!(!args.merge && !args.dry_run && !args.no_tensor_first_split);
        let args = parse(&["--split-max-size", "2G", "in.gguf", "out"]);
        assert_eq!(args.mode(), SplitMode::MaxBytes(2_000_000_000));
        let args = parse(&["--split-max-tensors", "7", "in.gguf", "out"]);
        assert_eq!(args.mode(), SplitMode::MaxTensors(7));
    }

    /// `gguf-split.cpp:119` and `:135`: the two operations and the two
    /// limits are each mutually exclusive.
    #[test]
    fn conflicting_flags_are_refused_at_parse_time() {
        for argv in [
            vec!["--split", "--merge", "in.gguf", "out"],
            vec![
                "--split-max-tensors",
                "3",
                "--split-max-size",
                "1G",
                "in.gguf",
                "out",
            ],
        ] {
            let mut full = vec!["gguf-split"];
            full.extend_from_slice(&argv);
            assert!(
                GgufSplitArgs::try_parse_from(&full).is_err(),
                "{argv:?} parsed"
            );
        }
    }

    /// Every split-only option has to be REFUSED by `--merge`, not
    /// parsed and dropped: llama.cpp accepts `--merge --split-max-size
    /// 1G` and ignores it. The list is walked rather than restated, so
    /// a new flag that `split_only_flags` classifies is covered here
    /// the moment it is added; the exhaustive destructure there is what
    /// stops a new flag from being classified as neither.
    #[test]
    fn every_split_only_flag_is_refused_by_merge_and_named_in_the_refusal() {
        let value_of = |flag: &str| match flag {
            "--split-max-tensors" => Some("4"),
            "--split-max-size" => Some("1G"),
            _ => None,
        };
        let flags = parse(&["in.gguf", "out"]).split_only_flags();
        assert_eq!(flags.len(), 3, "a flag was added without a case here");
        for (flag, typed) in flags {
            assert!(!typed, "`{flag}` reads as typed when it was not");
            let mut argv = vec!["--merge", flag];
            argv.extend(value_of(flag));
            argv.extend(["in.gguf", "out"]);
            let args = parse(&argv);
            assert!(
                args.split_only_flags()
                    .iter()
                    .any(|(name, typed)| *name == flag && *typed),
                "`{flag}` did not register as typed"
            );
            let err = run_merge(&args).unwrap_err().to_string();
            assert!(
                err.contains(flag),
                "merge with `{flag}` must name it; got: {err}"
            );
        }
    }
}
