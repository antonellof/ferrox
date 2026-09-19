//! CUDA graph capture/replay scaffolding for a per-token decode step.
//!
//! Pinned `cudarc` 0.11.9 exposes `cuGraph*` only via `driver::sys::lib()`
//! FFI (no safe wrapper). This module wraps the minimal capture →
//! instantiate → launch loop behind `FRINK_CUDA_GRAPH=1` so a Vast
//! re-measure can enable it without pulling a newer cudarc.
//!
//! Status: compiles under `--features cuda`. Capture requires a real
//! device and a caller that enqueues a fixed-shape decode into the
//! captured stream; see [`CudaDecodeGraph`]. Hardware receipt still
//! pending on comparable CUDA hardware.

use super::gpu::{shared_device, CudaError};
use std::sync::Mutex;

/// Whether CUDA-graph replay is requested (`FRINK_CUDA_GRAPH=1`).
pub fn cuda_graph_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("FRINK_CUDA_GRAPH").ok().as_deref(),
            Some("1") | Some("true") | Some("on")
        )
    })
}

/// Opaque handle for a captured decode graph. Empty until the first
/// successful capture on a live device; subsequent tokens can
/// [`CudaDecodeGraph::launch`] the instantiated exec.
pub struct CudaDecodeGraph {
    inner: Mutex<Option<GraphExec>>,
}

struct GraphExec {
    /// `CUgraphExec` as usize so this module stays Send without
    /// importing the raw pointer type into every call site.
    exec: usize,
}

// SAFETY: GraphExec is only touched while holding the Mutex, and CUDA
// graph exec objects are documented as usable from any thread of the
// creating context (frink uses one shared primary context).
unsafe impl Send for GraphExec {}

impl CudaDecodeGraph {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    pub fn has_exec(&self) -> bool {
        self.inner.lock().unwrap().is_some()
    }

    /// Begin stream capture on the null/legacy stream.
    /// Caller must enqueue the fixed-shape decode work, then call
    /// [`end_capture`].
    pub fn begin_capture(&self) -> Result<(), CudaError> {
        if !cuda_graph_enabled() {
            return Err(CudaError::Launch("FRINK_CUDA_GRAPH not enabled".into()));
        }
        let _dev = shared_device()?;
        // SAFETY: capturing the null stream is the cudarc-0.11 pattern
        // when no explicit stream handle is held; frink kernels use the
        // default stream.
        unsafe {
            use cudarc::driver::sys::{self, CUstreamCaptureMode};
            let err = sys::lib().cuStreamBeginCapture_v2(
                std::ptr::null_mut(),
                CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            );
            if err != sys::CUresult::CUDA_SUCCESS {
                return Err(CudaError::Launch(format!(
                    "cuStreamBeginCapture failed: {err:?}"
                )));
            }
        }
        Ok(())
    }

    /// End capture, instantiate, and store the exec for [`launch`].
    pub fn end_capture(&self) -> Result<(), CudaError> {
        let _dev = shared_device()?;
        unsafe {
            use cudarc::driver::sys;
            let mut graph: sys::CUgraph = std::ptr::null_mut();
            let err = sys::lib().cuStreamEndCapture(std::ptr::null_mut(), &mut graph);
            if err != sys::CUresult::CUDA_SUCCESS {
                return Err(CudaError::Launch(format!(
                    "cuStreamEndCapture failed: {err:?}"
                )));
            }
            let mut exec: sys::CUgraphExec = std::ptr::null_mut();
            let err = sys::lib().cuGraphInstantiate_v2(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            );
            let _ = sys::lib().cuGraphDestroy(graph);
            if err != sys::CUresult::CUDA_SUCCESS {
                return Err(CudaError::Launch(format!(
                    "cuGraphInstantiate failed: {err:?}"
                )));
            }
            *self.inner.lock().unwrap() = Some(GraphExec {
                exec: exec as usize,
            });
        }
        Ok(())
    }

    /// Replay the instantiated graph on the default stream.
    pub fn launch(&self) -> Result<(), CudaError> {
        let guard = self.inner.lock().unwrap();
        let Some(exec) = guard.as_ref() else {
            return Err(CudaError::Launch("no CUDA graph instantiated".into()));
        };
        unsafe {
            use cudarc::driver::sys;
            let err = sys::lib().cuGraphLaunch(exec.exec as sys::CUgraphExec, std::ptr::null_mut());
            if err != sys::CUresult::CUDA_SUCCESS {
                return Err(CudaError::Launch(format!("cuGraphLaunch failed: {err:?}")));
            }
        }
        Ok(())
    }
}

impl Default for CudaDecodeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CudaDecodeGraph {
    fn drop(&mut self) {
        if let Some(exec) = self.inner.lock().unwrap().take() {
            unsafe {
                use cudarc::driver::sys;
                let _ = sys::lib().cuGraphExecDestroy(exec.exec as sys::CUgraphExec);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_disabled_reports_no_exec() {
        let g = CudaDecodeGraph::new();
        assert!(!g.has_exec());
    }
}
