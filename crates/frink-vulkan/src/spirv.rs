//! A minimal SPIR-V binary emitter: enough to write one compute shader
//! by hand, and nothing more.
//!
//! # Why hand-emit SPIR-V
//!
//! The beachhead had to answer a question the roadmap asks explicitly:
//! *can frink's shaders be built without a C++ toolchain at build
//! time?* There are three ways to get SPIR-V into a Rust binary and
//! only one of them says yes without qualification:
//!
//! 1. **`glslangValidator` / `glslc` in `build.rs`.** This is what
//!    llama.cpp's Vulkan backend does (it builds a `vulkan-shaders-gen`
//!    C++ program at configure time). It makes a C++ toolchain a build
//!    prerequisite for every frink user who enables the feature, on
//!    every platform, forever. `frink-cuda` deliberately refused the
//!    equivalent (cudarc in dynamic-loading mode, no nvcc) and this
//!    crate refuses it for the same reason.
//! 2. **Commit a pre-built `.spv` blob.** No build dependency, but the
//!    repo then carries an opaque binary that no reviewer can read and
//!    that silently drifts from whatever source it was generated from.
//!    This codebase's whole test culture is against that.
//! 3. **Emit the words from Rust.** The shader is ordinary reviewable
//!    Rust, there is no build step at all, and the emitter is small
//!    enough to hold in your head. That is this file.
//!
//! The cost is stated honestly in the verdict: option 3 does not scale
//! to a hundred kernels. It scales to *one*, which is exactly the size
//! of a GO/NO-GO.
//!
//! # What this is not
//!
//! Not an SSA builder, not a validator, not a type deduplicator. It
//! writes words in the order it is told to, into the five sections
//! SPIR-V's logical layout requires. Getting the layout or the types
//! wrong produces an invalid module, and the crate's answer to that is
//! to run `spirv-val` over the emitted words in a test whenever the
//! tool is on `PATH` (see `q8_0_matvec`'s tests).

/// SPIR-V magic number (first word of every module).
pub const MAGIC: u32 = 0x0723_0203;

/// Target SPIR-V version, encoded `0 | major<<16 | minor<<8 | 0`.
///
/// **1.0 deliberately.** Vulkan 1.0 implementations are only required
/// to accept SPIR-V 1.0, and the beachhead's whole point is reaching
/// hardware frink cannot reach today -- which includes old Intel iGPUs
/// and whatever Android ships. Staying at 1.0 costs the `StorageBuffer`
/// storage class (this module uses the legacy `BufferBlock` + `Uniform`
/// spelling instead) and buys the widest possible device set.
pub const VERSION_1_0: u32 = 0x0001_0000;

/// Generator magic. 0 is "unregistered"; tools accept it.
pub const GENERATOR: u32 = 0;

// --- opcodes, only the ones used ---------------------------------

pub const OP_NAME: u16 = 5;
pub const OP_MEMBER_NAME: u16 = 6;
pub const OP_MEMORY_MODEL: u16 = 14;
pub const OP_ENTRY_POINT: u16 = 15;
pub const OP_EXECUTION_MODE: u16 = 16;
pub const OP_CAPABILITY: u16 = 17;
pub const OP_TYPE_VOID: u16 = 19;
pub const OP_TYPE_BOOL: u16 = 20;
pub const OP_TYPE_INT: u16 = 21;
pub const OP_TYPE_FLOAT: u16 = 22;
pub const OP_TYPE_VECTOR: u16 = 23;
pub const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
pub const OP_TYPE_STRUCT: u16 = 30;
pub const OP_TYPE_POINTER: u16 = 32;
pub const OP_TYPE_FUNCTION: u16 = 33;
pub const OP_CONSTANT: u16 = 43;
pub const OP_FUNCTION: u16 = 54;
pub const OP_FUNCTION_END: u16 = 56;
pub const OP_VARIABLE: u16 = 59;
pub const OP_LOAD: u16 = 61;
pub const OP_STORE: u16 = 62;
pub const OP_ACCESS_CHAIN: u16 = 65;
pub const OP_DECORATE: u16 = 71;
pub const OP_MEMBER_DECORATE: u16 = 72;
pub const OP_COMPOSITE_EXTRACT: u16 = 81;
pub const OP_CONVERT_S_TO_F: u16 = 111;
pub const OP_CONVERT_U_TO_F: u16 = 112;
pub const OP_BITCAST: u16 = 124;
pub const OP_F_NEGATE: u16 = 127;
pub const OP_I_ADD: u16 = 128;
pub const OP_F_ADD: u16 = 129;
pub const OP_I_SUB: u16 = 130;
pub const OP_I_MUL: u16 = 132;
pub const OP_F_MUL: u16 = 133;
pub const OP_SELECT: u16 = 169;
pub const OP_I_EQUAL: u16 = 170;
pub const OP_I_NOT_EQUAL: u16 = 171;
pub const OP_U_LESS_THAN: u16 = 176;
pub const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
pub const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
pub const OP_BITWISE_OR: u16 = 197;
pub const OP_BITWISE_XOR: u16 = 198;
pub const OP_BITWISE_AND: u16 = 199;
pub const OP_LOOP_MERGE: u16 = 246;
pub const OP_SELECTION_MERGE: u16 = 247;
pub const OP_LABEL: u16 = 248;
pub const OP_BRANCH: u16 = 249;
pub const OP_BRANCH_CONDITIONAL: u16 = 250;
pub const OP_RETURN: u16 = 253;

// --- enumerants, only the ones used ------------------------------

pub const CAP_SHADER: u32 = 1;
pub const ADDRESSING_LOGICAL: u32 = 0;
pub const MEMORY_MODEL_GLSL450: u32 = 1;
pub const EXEC_MODEL_GL_COMPUTE: u32 = 5;
pub const EXEC_MODE_LOCAL_SIZE: u32 = 17;

pub const SC_INPUT: u32 = 1;
pub const SC_UNIFORM: u32 = 2;
pub const SC_FUNCTION: u32 = 7;
pub const SC_PUSH_CONSTANT: u32 = 9;

pub const DEC_BLOCK: u32 = 2;
pub const DEC_BUFFER_BLOCK: u32 = 3;
pub const DEC_ARRAY_STRIDE: u32 = 6;
pub const DEC_BUILTIN: u32 = 11;
pub const DEC_BINDING: u32 = 33;
pub const DEC_DESCRIPTOR_SET: u32 = 34;
pub const DEC_OFFSET: u32 = 35;

pub const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

pub const NONE: u32 = 0;

/// Where an instruction goes in SPIR-V's required logical layout.
///
/// The order of the variants is the order of the sections in the
/// emitted module, and [`Builder::finish`] concatenates them in
/// declaration order -- so a section cannot be emitted out of place by
/// forgetting to sort.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Section {
    /// Capabilities, memory model, entry points, execution modes.
    Prelude,
    /// `OpName` / `OpMemberName`. Stripped by nothing here, but they
    /// make `spirv-dis` output readable, which is the only way to
    /// review a hand-built module.
    Debug,
    /// `OpDecorate` / `OpMemberDecorate`.
    Annotations,
    /// Types, constants, and every non-`Function` `OpVariable`.
    Types,
    /// Function definitions.
    Code,
}

/// Accumulates SPIR-V words per section and hands out result ids.
#[derive(Default)]
pub struct Builder {
    next_id: u32,
    prelude: Vec<u32>,
    debug: Vec<u32>,
    annotations: Vec<u32>,
    types: Vec<u32>,
    code: Vec<u32>,
}

impl Builder {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            ..Default::default()
        }
    }

    /// A fresh result `<id>`. Ids start at 1; 0 is never valid.
    pub fn id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn section(&mut self, s: Section) -> &mut Vec<u32> {
        match s {
            Section::Prelude => &mut self.prelude,
            Section::Debug => &mut self.debug,
            Section::Annotations => &mut self.annotations,
            Section::Types => &mut self.types,
            Section::Code => &mut self.code,
        }
    }

    /// Emit one instruction with no result id.
    pub fn inst(&mut self, s: Section, op: u16, operands: &[u32]) {
        let out = self.section(s);
        push_inst(out, op, operands);
    }

    /// Emit one instruction whose operands are `[result_type,
    /// result_id, ..rest]` and return `result_id`.
    pub fn typed(&mut self, s: Section, op: u16, result_type: u32, rest: &[u32]) -> u32 {
        let id = self.id();
        let mut ops = vec![result_type, id];
        ops.extend_from_slice(rest);
        self.inst(s, op, &ops);
        id
    }

    /// Emit one instruction whose only leading operand is its result id
    /// (`OpTypeInt`, `OpLabel`, ...) and return that id.
    pub fn result(&mut self, s: Section, op: u16, rest: &[u32]) -> u32 {
        let id = self.id();
        let mut ops = vec![id];
        ops.extend_from_slice(rest);
        self.inst(s, op, &ops);
        id
    }

    /// Emit an instruction that carries a trailing SPIR-V literal
    /// string (`OpName`, `OpEntryPoint`).
    pub fn inst_str(&mut self, s: Section, op: u16, head: &[u32], text: &str, tail: &[u32]) {
        let mut ops = head.to_vec();
        ops.extend(encode_string(text));
        ops.extend_from_slice(tail);
        self.inst(s, op, &ops);
    }

    /// The finished module: header followed by every section in
    /// layout order.
    pub fn finish(self) -> Vec<u32> {
        let mut out = Vec::with_capacity(
            5 + self.prelude.len()
                + self.debug.len()
                + self.annotations.len()
                + self.types.len()
                + self.code.len(),
        );
        out.push(MAGIC);
        out.push(VERSION_1_0);
        out.push(GENERATOR);
        // Bound: ids are `1..next_id`, and the bound is exclusive.
        out.push(self.next_id);
        out.push(0); // schema, reserved
        out.extend_from_slice(&self.prelude);
        out.extend_from_slice(&self.debug);
        out.extend_from_slice(&self.annotations);
        out.extend_from_slice(&self.types);
        out.extend_from_slice(&self.code);
        out
    }
}

fn push_inst(out: &mut Vec<u32>, op: u16, operands: &[u32]) {
    let word_count = operands.len() + 1;
    assert!(
        word_count < 0x1_0000,
        "SPIR-V instruction longer than the 16-bit word count field"
    );
    out.push(((word_count as u32) << 16) | op as u32);
    out.extend_from_slice(operands);
}

/// SPIR-V literal string: UTF-8, NUL-terminated, zero-padded to a whole
/// number of little-endian words.
pub fn encode_string(text: &str) -> Vec<u32> {
    let mut bytes = text.as_bytes().to_vec();
    bytes.push(0);
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    // `as_chunks` rather than `chunks_exact`: the padding above makes
    // the remainder provably empty, and clippy prefers the form that
    // hands back fixed-size arrays.
    let (words, rest) = bytes.as_chunks::<4>();
    debug_assert!(rest.is_empty(), "padding loop left a partial word");
    words.iter().map(|c| u32::from_le_bytes(*c)).collect()
}

/// The emitted words as the little-endian byte stream a `.spv` file
/// holds, for handing to an external validator.
pub fn to_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_well_formed_and_bound_covers_every_id() {
        let mut b = Builder::new();
        let a = b.id();
        let c = b.id();
        assert_eq!((a, c), (1, 2));
        let words = b.finish();
        assert_eq!(words[0], MAGIC);
        assert_eq!(words[1], VERSION_1_0);
        assert_eq!(words[3], 3, "bound must be one past the largest id");
        assert_eq!(words[4], 0);
        assert_eq!(words.len(), 5, "no sections were written");
    }

    #[test]
    fn instruction_word_count_is_operands_plus_one() {
        let mut b = Builder::new();
        b.inst(Section::Prelude, OP_CAPABILITY, &[CAP_SHADER]);
        let words = b.finish();
        assert_eq!(words[5], (2 << 16) | OP_CAPABILITY as u32);
        assert_eq!(words[6], CAP_SHADER);
    }

    #[test]
    fn sections_are_concatenated_in_layout_order_regardless_of_write_order() {
        let mut b = Builder::new();
        // Written backwards on purpose.
        b.inst(Section::Code, OP_RETURN, &[]);
        b.inst(Section::Annotations, OP_DECORATE, &[1, DEC_BLOCK]);
        b.inst(Section::Prelude, OP_CAPABILITY, &[CAP_SHADER]);
        let words = b.finish();
        let ops: Vec<u16> = decode_opcodes(&words[5..]);
        assert_eq!(ops, vec![OP_CAPABILITY, OP_DECORATE, OP_RETURN]);
    }

    #[test]
    fn strings_are_nul_terminated_and_word_padded() {
        // "main" is 4 bytes, so the NUL forces a second word.
        assert_eq!(encode_string("main").len(), 2);
        assert_eq!(encode_string("main")[1], 0);
        // 3 bytes + NUL fits exactly one word.
        assert_eq!(encode_string("abc").len(), 1);
        assert_eq!(encode_string("abc")[0], 0x0063_6261);
        assert_eq!(encode_string("").len(), 1);
        assert_eq!(encode_string("")[0], 0);
    }

    #[test]
    fn to_bytes_is_little_endian() {
        assert_eq!(to_bytes(&[MAGIC]), vec![0x03, 0x02, 0x23, 0x07]);
    }

    /// Split a word stream into opcodes by walking the word counts.
    /// Shared with `q8_0_matvec`'s structural tests.
    pub(crate) fn decode_opcodes(words: &[u32]) -> Vec<u16> {
        let mut ops = Vec::new();
        let mut i = 0;
        while i < words.len() {
            let count = (words[i] >> 16) as usize;
            assert!(count > 0, "zero-length instruction at word {i}");
            ops.push((words[i] & 0xffff) as u16);
            i += count;
        }
        assert_eq!(i, words.len(), "instruction stream overran the module");
        ops
    }
}
