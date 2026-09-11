//! The importance-matrix file, in both spellings llama.cpp b7650
//! reads and writes.
//!
//! **GGUF** (`tools/imatrix/imatrix.cpp:507-615`, `save_imatrix`) is
//! the current one, and the default `llama-imatrix` output since the
//! `.gguf` switch. A file is a GGUF whose metadata carries
//! `general.type = "imatrix"`, `imatrix.datasets` (string array),
//! `imatrix.chunk_count` (u32) and `imatrix.chunk_size` (u32), and
//! whose tensors come in pairs named after the weight they describe:
//! `<weight>.in_sum2`, F32 `[n_per_row, n_mat]`, the per-column sum of
//! squared activations; and `<weight>.counts`, F32 `[1, n_mat]`, how
//! many activation rows went into each matrix. `n_mat` is 1 for a dense
//! weight and the expert count for a MUL_MAT_ID weight. Names are
//! written in sorted order, `in_sum2` before `counts` for each. The
//! reader (`tools/quantize/quantize.cpp:218-330`, `load_imatrix`)
//! divides each sum by its count, and substitutes 1.0 for a matrix
//! whose count is zero.
//!
//! **Legacy** (`imatrix.cpp:401-505`, `save_imatrix_legacy`, written by
//! `--output-format dat`) is the pre-GGUF binary: an `int32` entry
//! count, then per entry an `int32` name length, the name, an `int32`
//! `ncall` (the entry's chunk count, rounded up), an `int32` value
//! count and that many `f32`s holding `(sum / count) * ncall`; then an
//! `int32` total chunk count and a length-prefixed dataset path. The
//! quantize reader (`quantize.cpp:152-216`) divides each value by
//! `ncall`. That round trip through `* ncall / ncall` is why a legacy
//! file's weights are not bit-identical to the GGUF file's for the same
//! run, and why this module reproduces the division rather than
//! "simplifying" it away.
//!
//! Both are read here; both are written by `ferrox imatrix`. The reader
//! tells them apart the way `quantize.cpp` does: a GGUF magic means
//! GGUF, anything else is tried as legacy.

use std::collections::BTreeMap;
use std::io::{BufWriter, Read};
use std::path::Path;

use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ferrox_gguf::{GgmlType, GgufFile, GgufValue, GgufWriter, TensorPlan, GGUF_MAGIC};

/// `general.type` value and the three `imatrix.*` keys, spelled once.
pub const KV_GENERAL_TYPE: &str = "general.type";
pub const GENERAL_TYPE_IMATRIX: &str = "imatrix";
pub const KV_DATASETS: &str = "imatrix.datasets";
pub const KV_CHUNK_COUNT: &str = "imatrix.chunk_count";
pub const KV_CHUNK_SIZE: &str = "imatrix.chunk_size";
const SUFFIX_IN_SUM2: &str = ".in_sum2";
const SUFFIX_COUNTS: &str = ".counts";

/// One weight's accumulated statistics: llama.cpp's `Stats`
/// (`imatrix.cpp:40-43`).
///
/// `values` is `n_per_row * n_mat` sums of squared activations, one
/// matrix after another; `counts` is one row count per matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Stats {
    pub values: Vec<f32>,
    pub counts: Vec<i64>,
}

impl Stats {
    pub fn n_mat(&self) -> usize {
        self.counts.len()
    }

    pub fn n_per_row(&self) -> usize {
        self.values.len() / self.n_mat().max(1)
    }
}

/// What `ferrox quantize --imatrix` consumes: per weight name, one
/// importance weight per column (times the expert count for a stacked
/// expert tensor), already normalised by the row count.
#[derive(Debug, Default, Clone)]
pub struct ImatrixWeights {
    pub entries: BTreeMap<String, Vec<f32>>,
    pub datasets: Vec<String>,
    /// `imatrix.chunk_count`, or the legacy trailer's `m_last_call`.
    /// Zero for a legacy file without the trailer.
    pub chunk_count: u32,
}

/// Reads either format. Errors name what is wrong with the file rather
/// than falling back to "no imatrix", because a quantize run that
/// silently proceeds unweighted is a worse file than none.
pub fn read(path: &Path) -> Result<ImatrixWeights> {
    let mut magic = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .with_context(|| format!("opening imatrix {}", path.display()))?;
    if u32::from_le_bytes(magic) == GGUF_MAGIC {
        read_gguf(path)
    } else {
        read_legacy(path)
    }
}

fn read_gguf(path: &Path) -> Result<ImatrixWeights> {
    let file =
        GgufFile::open(path).with_context(|| format!("parsing imatrix {}", path.display()))?;
    if file.metadata_str(KV_GENERAL_TYPE) != Some(GENERAL_TYPE_IMATRIX) {
        bail!(
            "{} is a GGUF but not an importance matrix: `{KV_GENERAL_TYPE}` is {:?}, expected \
             \"{GENERAL_TYPE_IMATRIX}\"",
            path.display(),
            file.metadata_str(KV_GENERAL_TYPE)
        );
    }
    // llama.cpp refuses a file missing any of the three keys
    // (`quantize.cpp:238-246`), so ferrox does too: a hand-edited file
    // that dropped one is a file whose provenance is unknown.
    let chunk_count = file
        .metadata_u64(KV_CHUNK_COUNT)
        .with_context(|| format!("{} has no `{KV_CHUNK_COUNT}`", path.display()))?
        as u32;
    file.metadata_u64(KV_CHUNK_SIZE)
        .with_context(|| format!("{} has no `{KV_CHUNK_SIZE}`", path.display()))?;
    let datasets = match file.metadata.get(KV_DATASETS) {
        Some(GgufValue::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .with_context(|| format!("`{KV_DATASETS}` holds a non-string entry"))
            })
            .collect::<Result<Vec<_>>>()?,
        _ => bail!("{} has no `{KV_DATASETS}` string array", path.display()),
    };

    // Pair `<name>.in_sum2` with `<name>.counts`. A name with one half
    // and not the other is refused, as upstream does.
    let mut sums: BTreeMap<String, &ferrox_gguf::TensorInfo> = BTreeMap::new();
    let mut counts: BTreeMap<String, &ferrox_gguf::TensorInfo> = BTreeMap::new();
    for t in &file.tensors {
        if let Some(name) = t.name.strip_suffix(SUFFIX_IN_SUM2) {
            sums.insert(name.to_string(), t);
        } else if let Some(name) = t.name.strip_suffix(SUFFIX_COUNTS) {
            counts.insert(name.to_string(), t);
        }
    }
    let mut entries = BTreeMap::new();
    for (name, sum_info) in &sums {
        let Some(count_info) = counts.get(name) else {
            bail!("mismatched sums and counts for {name}: `.in_sum2` without `.counts`");
        };
        let sum = f32_tensor(&file, sum_info)?;
        let cnt = f32_tensor(&file, count_info)?;
        let n_mat = cnt.len();
        if n_mat == 0 || !sum.len().is_multiple_of(n_mat) {
            bail!(
                "{name}: {} sums cannot be split over {n_mat} count(s)",
                sum.len()
            );
        }
        let ne0 = sum.len() / n_mat;
        let mut e = vec![0f32; sum.len()];
        for j in 0..n_mat {
            let count = cnt[j];
            for i in 0..ne0 {
                // `quantize.cpp:293-302`: divide by the count, or 1.0
                // for a matrix that saw no data.
                e[j * ne0 + i] = if count > 0.0 {
                    sum[j * ne0 + i] / count
                } else {
                    1.0
                };
            }
        }
        entries.insert(name.clone(), e);
    }
    for name in counts.keys() {
        if !sums.contains_key(name) {
            bail!("mismatched sums and counts for {name}: `.counts` without `.in_sum2`");
        }
    }
    if entries.is_empty() {
        bail!("no data in imatrix {}", path.display());
    }
    Ok(ImatrixWeights {
        entries,
        datasets,
        chunk_count,
    })
}

fn f32_tensor(file: &GgufFile, info: &ferrox_gguf::TensorInfo) -> Result<Vec<f32>> {
    if info.dtype != GgmlType::F32 {
        bail!(
            "imatrix tensor {} is {:?}; llama.cpp writes F32",
            info.name,
            info.dtype
        );
    }
    let bytes = file.tensor_bytes(&info.name)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn read_legacy(path: &Path) -> Result<ImatrixWeights> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut cur = std::io::Cursor::new(bytes.as_slice());
    let n_entries = cur
        .read_i32::<LittleEndian>()
        .context("legacy imatrix: entry count")?;
    if n_entries < 1 {
        bail!("no data in imatrix {} (legacy format)", path.display());
    }
    let mut entries = BTreeMap::new();
    for i in 0..n_entries {
        // Every length is bounded by the bytes still unread: a name or
        // a value array cannot be longer than the file that holds it.
        let remaining = |pos: u64| bytes.len().saturating_sub(pos as usize);
        let len = cur
            .read_i32::<LittleEndian>()
            .context("legacy imatrix: name length")?;
        if len < 0 || len as usize > remaining(cur.position()) {
            bail!("legacy imatrix entry {i}: name length {len} exceeds the file");
        }
        let mut name = vec![0u8; len as usize];
        cur.read_exact(&mut name)?;
        let name = String::from_utf8(name).context("legacy imatrix: name is not UTF-8")?;
        let ncall = cur
            .read_i32::<LittleEndian>()
            .context("legacy imatrix: ncall")?;
        let nval = cur
            .read_i32::<LittleEndian>()
            .context("legacy imatrix: nval")?;
        if nval < 1 || (nval as usize).saturating_mul(4) > remaining(cur.position()) {
            bail!("legacy imatrix entry {i} ({name}): value count {nval} exceeds the file");
        }
        let mut vals = vec![0f32; nval as usize];
        cur.read_f32_into::<LittleEndian>(&mut vals)?;
        if ncall > 0 {
            // `quantize.cpp:192-196`: `v /= ncall`, ncall as a float.
            for v in &mut vals {
                *v /= ncall as f32;
            }
        }
        entries.insert(name, vals);
    }
    // The trailer is optional in the oldest files; `quantize.cpp:203`
    // checks for EOF before reading it.
    let mut chunk_count = 0u32;
    let mut datasets = Vec::new();
    if (cur.position() as usize) < bytes.len() {
        chunk_count = cur.read_i32::<LittleEndian>().unwrap_or(0).max(0) as u32;
        if let Ok(len) = cur.read_i32::<LittleEndian>() {
            let remaining = bytes.len().saturating_sub(cur.position() as usize);
            if len > 0 && len as usize <= remaining {
                let mut d = vec![0u8; len as usize];
                cur.read_exact(&mut d)?;
                datasets.push(String::from_utf8_lossy(&d).into_owned());
            }
        }
    }
    Ok(ImatrixWeights {
        entries,
        datasets,
        chunk_count,
    })
}

/// How `ferrox imatrix` spells its output. Same two words as
/// `llama-imatrix --output-format`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Gguf,
    Dat,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "gguf" => Ok(OutputFormat::Gguf),
            "dat" => Ok(OutputFormat::Dat),
            other => Err(format!(
                "--output-format must be `gguf` or `dat`, not `{other}`"
            )),
        }
    }
}

/// Writes the collected statistics in the requested format.
pub fn write(
    path: &Path,
    format: OutputFormat,
    stats: &BTreeMap<String, Stats>,
    datasets: &[String],
    chunk_count: u32,
    chunk_size: u32,
) -> Result<()> {
    match format {
        OutputFormat::Gguf => write_gguf(path, stats, datasets, chunk_count, chunk_size),
        OutputFormat::Dat => write_legacy(path, stats, datasets, chunk_count, chunk_size),
    }
}

/// ggml trims trailing unit dimensions when it writes a tensor header
/// (`gguf_add_tensor` uses `ggml_n_dims`), so a dense weight's
/// `in_sum2` is written 1-D and its `counts` as `[1]`, while an expert
/// stack's are 2-D. Reproducing that keeps a ferrox file's headers
/// identical to llama.cpp's, not merely equivalent.
fn ggml_shape(ne0: usize, ne1: usize) -> Vec<u64> {
    if ne1 == 1 {
        vec![ne0 as u64]
    } else {
        vec![ne0 as u64, ne1 as u64]
    }
}

fn write_gguf(
    path: &Path,
    stats: &BTreeMap<String, Stats>,
    datasets: &[String],
    chunk_count: u32,
    chunk_size: u32,
) -> Result<()> {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        KV_GENERAL_TYPE.to_string(),
        GgufValue::String(GENERAL_TYPE_IMATRIX.into()),
    );
    // The writer refuses an empty array (it could not pick an element
    // type tag), and llama.cpp never writes one: a run always has a
    // prompt file or an input imatrix to name.
    metadata.insert(
        KV_DATASETS.to_string(),
        GgufValue::Array(
            datasets
                .iter()
                .map(|d| GgufValue::String(d.clone()))
                .collect(),
        ),
    );
    metadata.insert(KV_CHUNK_COUNT.to_string(), GgufValue::U32(chunk_count));
    metadata.insert(KV_CHUNK_SIZE.to_string(), GgufValue::U32(chunk_size));

    // `save_imatrix` writes every entry, partial or not; the reader
    // substitutes 1.0 for a zero count. `BTreeMap` iteration is the
    // sorted order upstream gets from `std::sort`.
    let mut plan = Vec::with_capacity(stats.len() * 2);
    for (name, s) in stats {
        let (n_mat, ne0) = (s.n_mat(), s.n_per_row());
        if s.values.is_empty() || n_mat == 0 {
            continue;
        }
        plan.push(TensorPlan {
            name: format!("{name}{SUFFIX_IN_SUM2}"),
            shape: ggml_shape(ne0, n_mat),
            dtype: GgmlType::F32,
            byte_len: s.values.len() * 4,
        });
        plan.push(TensorPlan {
            name: format!("{name}{SUFFIX_COUNTS}"),
            shape: ggml_shape(1, n_mat),
            dtype: GgmlType::F32,
            byte_len: n_mat * 4,
        });
    }
    let out =
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = GgufWriter::create(BufWriter::new(out), &metadata, plan)?;
    for (name, s) in stats {
        if s.values.is_empty() || s.n_mat() == 0 {
            continue;
        }
        let sums: Vec<u8> = s.values.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.write_tensor(&format!("{name}{SUFFIX_IN_SUM2}"), &sums)?;
        let counts: Vec<u8> = s
            .counts
            .iter()
            .flat_map(|c| (*c as f32).to_le_bytes())
            .collect();
        w.write_tensor(&format!("{name}{SUFFIX_COUNTS}"), &counts)?;
    }
    w.finish()?.into_inner().context("flushing")?;
    Ok(())
}

fn write_legacy(
    path: &Path,
    stats: &BTreeMap<String, Stats>,
    datasets: &[String],
    chunk_count: u32,
    chunk_size: u32,
) -> Result<()> {
    // `save_imatrix_legacy` SKIPS an entry with no data at all, unlike
    // the GGUF writer, and stores 1.0 for a matrix with a zero count
    // inside an entry that has some.
    let to_store: Vec<(&String, &Stats)> = stats
        .iter()
        .filter(|(_, s)| !s.counts.is_empty() && s.counts.iter().any(|&c| c != 0))
        .collect();
    let out =
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(out);
    w.write_i32::<LittleEndian>(to_store.len() as i32)?;
    for (name, s) in &to_store {
        w.write_i32::<LittleEndian>(name.len() as i32)?;
        w.write_all_bytes(name.as_bytes())?;
        let max_count = s.counts.iter().copied().max().unwrap_or(0);
        // Ceiling division, "to avoid accidental zeros".
        let ncall = ((max_count + i64::from(chunk_size) - 1) / i64::from(chunk_size)) as i32;
        w.write_i32::<LittleEndian>(ncall)?;
        let nval = s.values.len();
        let nmat = s.counts.len();
        w.write_i32::<LittleEndian>(nval as i32)?;
        for i in 0..nval {
            let mut count = s.counts[i / (nval / nmat)] as f32;
            let mut value = s.values[i];
            if count == 0.0 {
                value = 1.0;
                count = 1.0;
            }
            w.write_f32::<LittleEndian>((value / count) * ncall as f32)?;
        }
    }
    w.write_i32::<LittleEndian>(chunk_count as i32)?;
    let dataset = datasets.last().map(String::as_str).unwrap_or("");
    w.write_i32::<LittleEndian>(dataset.len() as i32)?;
    w.write_all_bytes(dataset.as_bytes())?;
    use std::io::Write;
    w.flush()?;
    Ok(())
}

/// `Write::write_all` under a name that does not collide with
/// `WriteBytesExt`'s methods in the calls above.
trait WriteAllBytes {
    fn write_all_bytes(&mut self, b: &[u8]) -> std::io::Result<()>;
}
impl<W: std::io::Write> WriteAllBytes for W {
    fn write_all_bytes(&mut self, b: &[u8]) -> std::io::Result<()> {
        self.write_all(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ferrox-imatrix-file-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sample() -> BTreeMap<String, Stats> {
        let mut m = BTreeMap::new();
        m.insert(
            "blk.0.attn_q.weight".to_string(),
            Stats {
                values: vec![10.0, 20.0, 30.0, 40.0],
                counts: vec![4],
            },
        );
        // A two-expert stack with one expert never routed to.
        m.insert(
            "blk.0.ffn_gate_exps.weight".to_string(),
            Stats {
                values: vec![6.0, 9.0, 0.0, 0.0],
                counts: vec![3, 0],
            },
        );
        m
    }

    /// The GGUF round trip: what `ferrox quantize` reads back is
    /// sum/count per column, 1.0 for the expert that saw nothing, and
    /// the metadata llama.cpp's reader insists on is all present.
    #[test]
    fn gguf_round_trip_divides_sums_by_counts_and_substitutes_one_for_empty() {
        let dir = tmp("gguf");
        let p = dir.join("im.gguf");
        write(
            &p,
            OutputFormat::Gguf,
            &sample(),
            &["calib.txt".to_string()],
            7,
            512,
        )
        .unwrap();
        let back = read(&p).unwrap();
        assert_eq!(back.chunk_count, 7);
        assert_eq!(back.datasets, vec!["calib.txt".to_string()]);
        assert_eq!(
            back.entries["blk.0.attn_q.weight"],
            vec![2.5, 5.0, 7.5, 10.0]
        );
        assert_eq!(
            back.entries["blk.0.ffn_gate_exps.weight"],
            vec![2.0, 3.0, 1.0, 1.0]
        );
        // The header shapes are ggml's trimmed ones.
        let f = GgufFile::open(&p).unwrap();
        assert_eq!(
            f.find_tensor("blk.0.attn_q.weight.in_sum2").unwrap().shape,
            vec![4]
        );
        assert_eq!(
            f.find_tensor("blk.0.attn_q.weight.counts").unwrap().shape,
            vec![1]
        );
        assert_eq!(
            f.find_tensor("blk.0.ffn_gate_exps.weight.in_sum2")
                .unwrap()
                .shape,
            vec![2, 2]
        );
        assert_eq!(
            f.find_tensor("blk.0.ffn_gate_exps.weight.counts")
                .unwrap()
                .shape,
            vec![1, 2]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The legacy round trip goes through `* ncall / ncall`, so it is
    /// pinned to the bytes `quantize.cpp` would compute, not to the
    /// exact quotient. The chunk size is 2 so `ncall` is 2 for the
    /// dense entry (`ceil(4 / 2)`) and the stored values are doubled;
    /// a reader that forgot the division would return `[5, 10, 15,
    /// 20]`. The zero-count expert comes back as exactly 1.0 because
    /// the writer stores 1.0 with a count of 1 and its `ncall` is 2
    /// (`ceil(3 / 2)`), so `1 * 2 / 2`.
    #[test]
    fn legacy_round_trip_reproduces_quantize_cpps_division() {
        let dir = tmp("dat");
        let p = dir.join("im.dat");
        write(
            &p,
            OutputFormat::Dat,
            &sample(),
            &["calib.txt".to_string()],
            7,
            2,
        )
        .unwrap();
        let back = read(&p).unwrap();
        assert_eq!(back.chunk_count, 7);
        assert_eq!(back.datasets, vec!["calib.txt".to_string()]);
        assert_eq!(
            back.entries["blk.0.attn_q.weight"],
            vec![2.5, 5.0, 7.5, 10.0]
        );
        assert_eq!(
            back.entries["blk.0.ffn_gate_exps.weight"],
            vec![2.0, 3.0, 1.0, 1.0]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A GGUF that is not an imatrix is refused by name, not read as
    /// an empty one: `general.type` is the discriminator llama.cpp
    /// writes, and a model file has 300 tensors none of which end in
    /// `.in_sum2`.
    #[test]
    fn a_gguf_without_the_imatrix_type_is_refused() {
        let dir = tmp("notim");
        let p = dir.join("model.gguf");
        let mut md = BTreeMap::new();
        md.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".into()),
        );
        let f = std::fs::File::create(&p).unwrap();
        GgufWriter::create(BufWriter::new(f), &md, vec![])
            .unwrap()
            .finish()
            .unwrap();
        let err = read(&p).unwrap_err().to_string();
        assert!(err.contains("not an importance matrix"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A legacy header claiming more values than the file holds is
    /// refused before any allocation of that size. `i32::MAX` values
    /// is 8 GiB, which is the size the loop would otherwise ask for.
    #[test]
    fn a_legacy_value_count_larger_than_the_file_is_refused_before_allocating() {
        let dir = tmp("bound");
        let p = dir.join("bad.dat");
        let mut b = Vec::new();
        b.write_i32::<LittleEndian>(1).unwrap();
        b.write_i32::<LittleEndian>(3).unwrap();
        b.extend_from_slice(b"abc");
        b.write_i32::<LittleEndian>(1).unwrap();
        b.write_i32::<LittleEndian>(i32::MAX).unwrap();
        std::fs::write(&p, &b).unwrap();
        let err = read(&p).unwrap_err().to_string();
        assert!(err.contains("exceeds the file"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
