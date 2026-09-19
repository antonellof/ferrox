//! A minimal row-major dense f32 tensor. Frink keeps this deliberately
//! small: weights live quantized in the mmap'd GGUF file and are
//! dequantized on demand by frink-quant; this type is for activations
//! and small dequantized weight slices during the forward pass.

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "tensor data length {} does not match shape {:?} (expected {})",
            data.len(),
            shape,
            expected
        );
        Tensor { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; n],
            shape,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    pub fn cols(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(1)
    }

    pub fn row(&self, i: usize) -> &[f32] {
        let cols = self.cols();
        &self.data[i * cols..(i + 1) * cols]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_indexing_matches_shape() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        assert_eq!(t.row(0), &[1.0, 2.0]);
        assert_eq!(t.row(1), &[3.0, 4.0]);
        assert_eq!(t.row(2), &[5.0, 6.0]);
    }

    #[test]
    #[should_panic]
    fn mismatched_shape_panics() {
        Tensor::new(vec![1.0, 2.0, 3.0], vec![2, 2]);
    }
}
