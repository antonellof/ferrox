//! One place that says which safetensors element types widen to `f32`
//! and how.
//!
//! Two loaders in this crate read safetensors weights into `f32`
//! (`kimi_loader` for a whole decoder, `rerank_pooler` for one BERT
//! pooler), and before this module each carried its own `match` over
//! [`SafetensorsDtype`]. Two matches that must agree about which dtypes
//! are readable is this repo's dominant bug shape, so the dispatch lives
//! here once and both call it.
//!
//! Only the *lossless* widenings are here: F32 verbatim, F16 and BF16
//! through `ferrox_quant`'s widening functions. Anything else is `None`
//! and the caller names the tensor in its refusal; an integer or f64
//! tensor is not something a weight loader should quietly cast.

use ferrox_safetensors::SafetensorsDtype;

/// `raw` as `f32`, or `None` when `dtype` is not one of the three
/// float types this crate reads. `None` for a length that is not a
/// multiple of the element width too, which `ferrox-safetensors` has
/// already refused at parse time, so it is stated rather than relied on.
pub(crate) fn widen_to_f32(dtype: SafetensorsDtype, raw: &[u8]) -> Option<Vec<f32>> {
    match dtype {
        SafetensorsDtype::F32 => {
            let (chunks, rest) = raw.as_chunks::<4>();
            if !rest.is_empty() {
                return None;
            }
            Some(chunks.iter().map(|c| f32::from_le_bytes(*c)).collect())
        }
        SafetensorsDtype::F16 => ferrox_quant::dequant_f16(raw).ok(),
        SafetensorsDtype::BF16 => ferrox_quant::dequant_bf16(raw).ok(),
        SafetensorsDtype::Bool
        | SafetensorsDtype::U8
        | SafetensorsDtype::I8
        | SafetensorsDtype::I16
        | SafetensorsDtype::U16
        | SafetensorsDtype::I32
        | SafetensorsDtype::U32
        | SafetensorsDtype::I64
        | SafetensorsDtype::U64
        | SafetensorsDtype::F64 => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three float widths agree on the same value, and the
    /// integer widths are refused rather than reinterpreted.
    #[test]
    fn the_three_float_widths_widen_losslessly_and_nothing_else_widens() {
        let value = 1.5f32;
        let f32_bytes = value.to_le_bytes();
        let f16_bytes = half::f16::from_f32(value).to_le_bytes();
        let bf16_bytes = (value.to_bits() >> 16) as u16;
        assert_eq!(
            widen_to_f32(SafetensorsDtype::F32, &f32_bytes),
            Some(vec![1.5])
        );
        assert_eq!(
            widen_to_f32(SafetensorsDtype::F16, &f16_bytes),
            Some(vec![1.5])
        );
        assert_eq!(
            widen_to_f32(SafetensorsDtype::BF16, &bf16_bytes.to_le_bytes()),
            Some(vec![1.5])
        );
        assert_eq!(widen_to_f32(SafetensorsDtype::I32, &f32_bytes), None);
        assert_eq!(widen_to_f32(SafetensorsDtype::F64, &[0u8; 8]), None);
    }

    /// A byte length that is not a whole number of elements is `None`,
    /// not a truncated vector: a tensor missing its last element is not
    /// the tensor.
    #[test]
    fn a_ragged_byte_length_is_refused_rather_than_truncated() {
        assert_eq!(widen_to_f32(SafetensorsDtype::F32, &[0u8; 6]), None);
        assert_eq!(widen_to_f32(SafetensorsDtype::F16, &[0u8; 3]), None);
    }
}
