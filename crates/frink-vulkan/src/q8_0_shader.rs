//! The beachhead kernel: a Q8_0 matvec, emitted as a SPIR-V compute
//! shader with no external shader compiler.
//!
//! # The shape, and why it is this shape
//!
//! One invocation per output row, `y[row] = dot(dequant(W[row]), x)`.
//! That is deliberately the *slowest* correct shape -- no subgroup
//! reduction, no shared-memory tiling, no `dp4a`-style integer dot. The
//! beachhead's question is "can frink reach a Vulkan device, upload a
//! quantized weight, run a shader, and read back a correct answer",
//! and a fast kernel answers it no better than a slow one while being
//! far harder to hand-emit and to check. **Nothing here is a
//! performance claim.**
//!
//! # The one thing that surprised, and it generalises
//!
//! A ggml Q8_0 block is **34 bytes** (an f16 scale plus 32 int8s), and
//! 34 is not a multiple of 4. A row of `n` blocks is `34n` bytes, which
//! is 4-byte aligned only when `n` is even. SPIR-V's logical addressing
//! has no byte pointer: a storage buffer is an array of some type, and
//! the smallest type available without the `Int8`/`StorageBuffer8Bit`
//! capabilities (which old Intel and Android drivers may not have) is
//! `uint`. So every byte of every weight is reached as
//! `(w[k >> 2] >> ((k & 3) * 8)) & 0xff`, and the f16 scale is decoded
//! with integer bit arithmetic rather than a hardware `float16_t`.
//!
//! That is not a quirk of Q8_0. Q4_0 is 18 bytes, Q4_K is 144, Q6_K is
//! 210 -- ggml's block sizes are simply not word multiples, so *any*
//! Vulkan backend for frink either does this byte extraction
//! everywhere or repacks weights on upload and gives up the
//! zero-copy-from-mmap property that `amd-strix-halo` built its whole
//! UMA argument on. llama.cpp's Vulkan backend takes the first road.
//! This kernel takes it too, so that the cost is measured rather than
//! assumed. See the verdict for what it costs.
//!
//! # Descriptor layout
//!
//! | set | binding | contents |
//! |---|---|---|
//! | 0 | 0 | `uint[]` -- the Q8_0 rows, verbatim GGUF bytes |
//! | 0 | 1 | `float[]` -- the activation `x`, `cols` entries |
//! | 0 | 2 | `float[]` -- the output `y`, `rows` entries |
//!
//! Push constants, 12 bytes: `rows`, `n_blocks_per_row`, `row_bytes`.
//! `row_bytes` is redundant (`34 * n_blocks_per_row`) and is passed
//! anyway, because that is the argument list
//! `weight_matrix.rs`'s `CudaMatvecLaunchFn` already carries and the
//! seam survey recommends unifying on.

use crate::spirv::*;

/// Elements per Q8_0 block.
pub const BLOCK_ELEMS: usize = 32;
/// Bytes per Q8_0 block: an f16 scale followed by 32 int8 weights.
pub const BLOCK_BYTES: usize = 34;
/// Invocations per workgroup. One row each.
pub const LOCAL_SIZE_X: u32 = 64;
/// Entry point name the pipeline must ask for.
pub const ENTRY_POINT: &str = "main";

/// `2^-24`, the f16 subnormal quantum. Exact in f32.
const F16_SUBNORMAL_SCALE_BITS: u32 = 0x3380_0000;

/// Type and constant ids shared by every instruction in the module.
struct Ids {
    void: u32,
    bool_: u32,
    u32_: u32,
    i32_: u32,
    f32_: u32,
    ptr_uniform_u32: u32,
    ptr_uniform_f32: u32,
    ptr_pc_u32: u32,
    ptr_fn_u32: u32,
    ptr_fn_f32: u32,
    var_w: u32,
    var_x: u32,
    var_y: u32,
    var_pc: u32,
    var_gid: u32,
    v3u: u32,
    /// Cached `uint` constants, keyed by value.
    u_consts: Vec<(u32, u32)>,
    f_zero: u32,
    f_sub_scale: u32,
}

struct Kernel {
    b: Builder,
    i: Ids,
}

impl Kernel {
    /// A cached `OpConstant` of `u32` type.
    fn u(&mut self, v: u32) -> u32 {
        if let Some((_, id)) = self.i.u_consts.iter().find(|(k, _)| *k == v) {
            return *id;
        }
        let id = self.b.typed(Section::Types, OP_CONSTANT, self.i.u32_, &[v]);
        self.i.u_consts.push((v, id));
        id
    }

    /// One binary op in the function body.
    fn op2(&mut self, op: u16, ty: u32, a: u32, c: u32) -> u32 {
        self.b.typed(Section::Code, op, ty, &[a, c])
    }

    /// One unary op in the function body.
    fn op1(&mut self, op: u16, ty: u32, a: u32) -> u32 {
        self.b.typed(Section::Code, op, ty, &[a])
    }

    fn label(&mut self) -> u32 {
        self.b.result(Section::Code, OP_LABEL, &[])
    }

    fn branch(&mut self, target: u32) {
        self.b.inst(Section::Code, OP_BRANCH, &[target]);
    }

    /// `(w[k >> 2] >> ((k & 3) * 8)) & 0xff` -- one byte of the weight
    /// buffer, addressed by absolute byte index.
    fn weight_byte(&mut self, k: u32) -> u32 {
        let c0 = self.u(0);
        let c2 = self.u(2);
        let c3 = self.u(3);
        let c8 = self.u(8);
        let c255 = self.u(255);
        let word_index = self.op2(OP_SHIFT_RIGHT_LOGICAL, self.i.u32_, k, c2);
        let byte_in_word = self.op2(OP_BITWISE_AND, self.i.u32_, k, c3);
        let shift = self.op2(OP_I_MUL, self.i.u32_, byte_in_word, c8);
        let ptr = self.b.typed(
            Section::Code,
            OP_ACCESS_CHAIN,
            self.i.ptr_uniform_u32,
            &[self.i.var_w, c0, word_index],
        );
        let word = self.op1(OP_LOAD, self.i.u32_, ptr);
        let shifted = self.op2(OP_SHIFT_RIGHT_LOGICAL, self.i.u32_, word, shift);
        self.op2(OP_BITWISE_AND, self.i.u32_, shifted, c255)
    }

    /// Decode an IEEE binary16 held in the low 16 bits of `h` into an
    /// f32, using only integer ops and `OpBitcast`.
    ///
    /// No `Float16` capability, no `VK_KHR_16bit_storage`: those are
    /// optional in Vulkan 1.0 and the whole point of the beachhead is
    /// the devices frink cannot currently reach. `f16_to_f32` in
    /// `q8_0_reference` is the same arithmetic in Rust, and
    /// `f16_decode_matches_half_crate` holds it against the `half`
    /// crate over **every one of the 65,536 bit patterns**.
    fn decode_f16(&mut self, h: u32) -> u32 {
        let c0 = self.u(0);
        let c10 = self.u(10);
        let c13 = self.u(13);
        let c15 = self.u(15);
        let c23 = self.u(23);
        let c31 = self.u(31);
        let c112 = self.u(112);
        let c1023 = self.u(1023);
        let c_inf = self.u(0x7f80_0000);

        let sign = self.op2(OP_SHIFT_RIGHT_LOGICAL, self.i.u32_, h, c15);
        let exp_raw = self.op2(OP_SHIFT_RIGHT_LOGICAL, self.i.u32_, h, c10);
        let exp = self.op2(OP_BITWISE_AND, self.i.u32_, exp_raw, c31);
        let mant = self.op2(OP_BITWISE_AND, self.i.u32_, h, c1023);
        let mant_hi = self.op2(OP_SHIFT_LEFT_LOGICAL, self.i.u32_, mant, c13);

        // Normal: rebias 15 -> 127, i.e. + 112, then place the mantissa.
        let exp_f32 = self.op2(OP_I_ADD, self.i.u32_, exp, c112);
        let exp_bits = self.op2(OP_SHIFT_LEFT_LOGICAL, self.i.u32_, exp_f32, c23);
        let normal_bits = self.op2(OP_BITWISE_OR, self.i.u32_, exp_bits, mant_hi);
        let normal = self.op1(OP_BITCAST, self.i.f32_, normal_bits);

        // Subnormal (exp == 0): value is mant * 2^-24, exact in f32.
        let mant_f = self.op1(OP_CONVERT_U_TO_F, self.i.f32_, mant);
        let subnormal = self.op2(OP_F_MUL, self.i.f32_, mant_f, self.i.f_sub_scale);

        // exp == 31: infinity when mant == 0, NaN otherwise. Widening
        // the mantissa preserves both, and the quiet bit lands where
        // f32 wants it.
        let inf_bits = self.op2(OP_BITWISE_OR, self.i.u32_, c_inf, mant_hi);
        let inf_or_nan = self.op1(OP_BITCAST, self.i.f32_, inf_bits);

        let exp_is_zero = self.op2(OP_I_EQUAL, self.i.bool_, exp, c0);
        let exp_is_max = self.op2(OP_I_EQUAL, self.i.bool_, exp, c31);
        let pick = self.b.typed(
            Section::Code,
            OP_SELECT,
            self.i.f32_,
            &[exp_is_max, inf_or_nan, normal],
        );
        let magnitude = self.b.typed(
            Section::Code,
            OP_SELECT,
            self.i.f32_,
            &[exp_is_zero, subnormal, pick],
        );
        let negated = self.op1(OP_F_NEGATE, self.i.f32_, magnitude);
        let is_negative = self.op2(OP_I_NOT_EQUAL, self.i.bool_, sign, c0);
        self.b.typed(
            Section::Code,
            OP_SELECT,
            self.i.f32_,
            &[is_negative, negated, magnitude],
        )
    }
}

/// The complete SPIR-V module for the Q8_0 matvec, as words.
pub fn spirv() -> Vec<u32> {
    let mut b = Builder::new();

    // --- types -------------------------------------------------
    let void = b.result(Section::Types, OP_TYPE_VOID, &[]);
    let fn_void = b.result(Section::Types, OP_TYPE_FUNCTION, &[void]);
    let bool_ = b.result(Section::Types, OP_TYPE_BOOL, &[]);
    let u32_ = b.result(Section::Types, OP_TYPE_INT, &[32, 0]);
    let i32_ = b.result(Section::Types, OP_TYPE_INT, &[32, 1]);
    let f32_ = b.result(Section::Types, OP_TYPE_FLOAT, &[32]);
    let v3u = b.result(Section::Types, OP_TYPE_VECTOR, &[u32_, 3]);

    let arr_u32 = b.result(Section::Types, OP_TYPE_RUNTIME_ARRAY, &[u32_]);
    let arr_f32 = b.result(Section::Types, OP_TYPE_RUNTIME_ARRAY, &[f32_]);
    let struct_w = b.result(Section::Types, OP_TYPE_STRUCT, &[arr_u32]);
    let struct_f = b.result(Section::Types, OP_TYPE_STRUCT, &[arr_f32]);
    let struct_pc = b.result(Section::Types, OP_TYPE_STRUCT, &[u32_, u32_, u32_]);

    let ptr_uniform_w = b.result(Section::Types, OP_TYPE_POINTER, &[SC_UNIFORM, struct_w]);
    let ptr_uniform_f = b.result(Section::Types, OP_TYPE_POINTER, &[SC_UNIFORM, struct_f]);
    let ptr_pc = b.result(
        Section::Types,
        OP_TYPE_POINTER,
        &[SC_PUSH_CONSTANT, struct_pc],
    );
    let ptr_uniform_u32 = b.result(Section::Types, OP_TYPE_POINTER, &[SC_UNIFORM, u32_]);
    let ptr_uniform_f32 = b.result(Section::Types, OP_TYPE_POINTER, &[SC_UNIFORM, f32_]);
    let ptr_pc_u32 = b.result(Section::Types, OP_TYPE_POINTER, &[SC_PUSH_CONSTANT, u32_]);
    let ptr_in_v3u = b.result(Section::Types, OP_TYPE_POINTER, &[SC_INPUT, v3u]);
    let ptr_fn_u32 = b.result(Section::Types, OP_TYPE_POINTER, &[SC_FUNCTION, u32_]);
    let ptr_fn_f32 = b.result(Section::Types, OP_TYPE_POINTER, &[SC_FUNCTION, f32_]);

    let f_zero = b.typed(Section::Types, OP_CONSTANT, f32_, &[0]);
    let f_sub_scale = b.typed(
        Section::Types,
        OP_CONSTANT,
        f32_,
        &[F16_SUBNORMAL_SCALE_BITS],
    );

    // --- global variables --------------------------------------
    let var_w = b.typed(Section::Types, OP_VARIABLE, ptr_uniform_w, &[SC_UNIFORM]);
    let var_x = b.typed(Section::Types, OP_VARIABLE, ptr_uniform_f, &[SC_UNIFORM]);
    let var_y = b.typed(Section::Types, OP_VARIABLE, ptr_uniform_f, &[SC_UNIFORM]);
    let var_pc = b.typed(Section::Types, OP_VARIABLE, ptr_pc, &[SC_PUSH_CONSTANT]);
    let var_gid = b.typed(Section::Types, OP_VARIABLE, ptr_in_v3u, &[SC_INPUT]);

    let main = b.id();

    // --- prelude -----------------------------------------------
    b.inst(Section::Prelude, OP_CAPABILITY, &[CAP_SHADER]);
    b.inst(
        Section::Prelude,
        OP_MEMORY_MODEL,
        &[ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450],
    );
    // SPIR-V 1.0 lists only Input/Output variables in the interface.
    b.inst_str(
        Section::Prelude,
        OP_ENTRY_POINT,
        &[EXEC_MODEL_GL_COMPUTE, main],
        ENTRY_POINT,
        &[var_gid],
    );
    b.inst(
        Section::Prelude,
        OP_EXECUTION_MODE,
        &[main, EXEC_MODE_LOCAL_SIZE, LOCAL_SIZE_X, 1, 1],
    );

    // --- debug names -------------------------------------------
    for (id, name) in [
        (main, "q8_0_matvec"),
        (var_w, "weights"),
        (var_x, "x"),
        (var_y, "y"),
        (var_pc, "pc"),
    ] {
        b.inst_str(Section::Debug, OP_NAME, &[id], name, &[]);
    }
    for (member, name) in [(0u32, "rows"), (1, "n_blocks"), (2, "row_bytes")] {
        b.inst_str(
            Section::Debug,
            OP_MEMBER_NAME,
            &[struct_pc, member],
            name,
            &[],
        );
    }

    // --- annotations -------------------------------------------
    b.inst(
        Section::Annotations,
        OP_DECORATE,
        &[arr_u32, DEC_ARRAY_STRIDE, 4],
    );
    b.inst(
        Section::Annotations,
        OP_DECORATE,
        &[arr_f32, DEC_ARRAY_STRIDE, 4],
    );
    b.inst(
        Section::Annotations,
        OP_DECORATE,
        &[struct_w, DEC_BUFFER_BLOCK],
    );
    b.inst(
        Section::Annotations,
        OP_DECORATE,
        &[struct_f, DEC_BUFFER_BLOCK],
    );
    b.inst(
        Section::Annotations,
        OP_MEMBER_DECORATE,
        &[struct_w, 0, DEC_OFFSET, 0],
    );
    b.inst(
        Section::Annotations,
        OP_MEMBER_DECORATE,
        &[struct_f, 0, DEC_OFFSET, 0],
    );
    b.inst(Section::Annotations, OP_DECORATE, &[struct_pc, DEC_BLOCK]);
    for member in 0..3u32 {
        b.inst(
            Section::Annotations,
            OP_MEMBER_DECORATE,
            &[struct_pc, member, DEC_OFFSET, member * 4],
        );
    }
    for (var, binding) in [(var_w, 0u32), (var_x, 1), (var_y, 2)] {
        b.inst(
            Section::Annotations,
            OP_DECORATE,
            &[var, DEC_DESCRIPTOR_SET, 0],
        );
        b.inst(
            Section::Annotations,
            OP_DECORATE,
            &[var, DEC_BINDING, binding],
        );
    }
    b.inst(
        Section::Annotations,
        OP_DECORATE,
        &[var_gid, DEC_BUILTIN, BUILTIN_GLOBAL_INVOCATION_ID],
    );

    let mut k = Kernel {
        b,
        i: Ids {
            void,
            bool_,
            u32_,
            i32_,
            f32_,
            ptr_uniform_u32,
            ptr_uniform_f32,
            ptr_pc_u32,
            ptr_fn_u32,
            ptr_fn_f32,
            var_w,
            var_x,
            var_y,
            var_pc,
            var_gid,
            v3u,
            u_consts: Vec::new(),
            f_zero,
            f_sub_scale,
        },
    };
    emit_main(&mut k, main, fn_void);
    k.b.finish()
}

/// The function body. Split out only so no single function in this file
/// runs past a screenful of blocks.
fn emit_main(k: &mut Kernel, main: u32, fn_void: u32) {
    let (void, u32_, i32_, f32_, bool_) = (k.i.void, k.i.u32_, k.i.i32_, k.i.f32_, k.i.bool_);
    k.b.inst(Section::Code, OP_FUNCTION, &[void, main, NONE, fn_void]);

    let entry = k.label();
    // Every Function-storage OpVariable must be the first instructions
    // of the entry block.
    let acc =
        k.b.typed(Section::Code, OP_VARIABLE, k.i.ptr_fn_f32, &[SC_FUNCTION]);
    let blk =
        k.b.typed(Section::Code, OP_VARIABLE, k.i.ptr_fn_u32, &[SC_FUNCTION]);
    let elem =
        k.b.typed(Section::Code, OP_VARIABLE, k.i.ptr_fn_u32, &[SC_FUNCTION]);

    let c0 = k.u(0);
    let c1 = k.u(1);
    let c2 = k.u(2);
    let c8 = k.u(8);
    let c32 = k.u(BLOCK_ELEMS as u32);
    let c34 = k.u(BLOCK_BYTES as u32);
    let c128 = k.u(128);

    let gid = k.op1(OP_LOAD, k.i.v3u, k.i.var_gid);
    let row =
        k.b.typed(Section::Code, OP_COMPOSITE_EXTRACT, u32_, &[gid, 0]);
    let rows = load_push_constant(k, 0);
    let in_range = k.op2(OP_U_LESS_THAN, bool_, row, rows);

    let work = k.b.id();
    let exit = k.b.id();
    k.b.inst(Section::Code, OP_SELECTION_MERGE, &[exit, NONE]);
    k.b.inst(
        Section::Code,
        OP_BRANCH_CONDITIONAL,
        &[in_range, work, exit],
    );
    let _ = entry;

    // --- %work -------------------------------------------------
    k.b.inst(Section::Code, OP_LABEL, &[work]);
    k.b.inst(Section::Code, OP_STORE, &[acc, k.i.f_zero]);
    let n_blocks = load_push_constant(k, 1);
    let row_bytes = load_push_constant(k, 2);
    let row_base = k.op2(OP_I_MUL, u32_, row, row_bytes);
    k.b.inst(Section::Code, OP_STORE, &[blk, c0]);

    let outer_header = k.b.id();
    let outer_cond = k.b.id();
    let outer_body = k.b.id();
    let outer_cont = k.b.id();
    let outer_merge = k.b.id();
    k.branch(outer_header);

    k.b.inst(Section::Code, OP_LABEL, &[outer_header]);
    k.b.inst(
        Section::Code,
        OP_LOOP_MERGE,
        &[outer_merge, outer_cont, NONE],
    );
    k.branch(outer_cond);

    k.b.inst(Section::Code, OP_LABEL, &[outer_cond]);
    let b_now = k.op1(OP_LOAD, u32_, blk);
    let more_blocks = k.op2(OP_U_LESS_THAN, bool_, b_now, n_blocks);
    k.b.inst(
        Section::Code,
        OP_BRANCH_CONDITIONAL,
        &[more_blocks, outer_body, outer_merge],
    );

    // --- %outer_body: decode this block's f16 scale ------------
    k.b.inst(Section::Code, OP_LABEL, &[outer_body]);
    let b_i = k.op1(OP_LOAD, u32_, blk);
    let blk_off = k.op2(OP_I_MUL, u32_, b_i, c34);
    let off = k.op2(OP_I_ADD, u32_, row_base, blk_off);
    let lo = k.weight_byte(off);
    let off1 = k.op2(OP_I_ADD, u32_, off, c1);
    let hi = k.weight_byte(off1);
    let hi_shifted = k.op2(OP_SHIFT_LEFT_LOGICAL, u32_, hi, c8);
    let h = k.op2(OP_BITWISE_OR, u32_, lo, hi_shifted);
    let scale = k.decode_f16(h);
    let x_base = k.op2(OP_I_MUL, u32_, b_i, c32);
    let q_base = k.op2(OP_I_ADD, u32_, off, c2);
    k.b.inst(Section::Code, OP_STORE, &[elem, c0]);

    let inner_header = k.b.id();
    let inner_cond = k.b.id();
    let inner_body = k.b.id();
    let inner_cont = k.b.id();
    let inner_merge = k.b.id();
    k.branch(inner_header);

    k.b.inst(Section::Code, OP_LABEL, &[inner_header]);
    k.b.inst(
        Section::Code,
        OP_LOOP_MERGE,
        &[inner_merge, inner_cont, NONE],
    );
    k.branch(inner_cond);

    k.b.inst(Section::Code, OP_LABEL, &[inner_cond]);
    let j_now = k.op1(OP_LOAD, u32_, elem);
    let more_elems = k.op2(OP_U_LESS_THAN, bool_, j_now, c32);
    k.b.inst(
        Section::Code,
        OP_BRANCH_CONDITIONAL,
        &[more_elems, inner_body, inner_merge],
    );

    // --- %inner_body: one int8 weight times one activation -----
    k.b.inst(Section::Code, OP_LABEL, &[inner_body]);
    let j = k.op1(OP_LOAD, u32_, elem);
    let q_index = k.op2(OP_I_ADD, u32_, q_base, j);
    let q_byte = k.weight_byte(q_index);
    // Two's-complement sign extension without an Int8 capability:
    // (b ^ 0x80) - 0x80 wraps in uint exactly as int8 would.
    let flipped = k.op2(OP_BITWISE_XOR, u32_, q_byte, c128);
    let biased = k.op2(OP_I_SUB, u32_, flipped, c128);
    let signed = k.op1(OP_BITCAST, i32_, biased);
    let q_f = k.op1(OP_CONVERT_S_TO_F, f32_, signed);
    let x_index = k.op2(OP_I_ADD, u32_, x_base, j);
    let x_ptr = k.b.typed(
        Section::Code,
        OP_ACCESS_CHAIN,
        k.i.ptr_uniform_f32,
        &[k.i.var_x, c0, x_index],
    );
    let x_val = k.op1(OP_LOAD, f32_, x_ptr);
    let scaled = k.op2(OP_F_MUL, f32_, scale, q_f);
    let term = k.op2(OP_F_MUL, f32_, scaled, x_val);
    let acc_old = k.op1(OP_LOAD, f32_, acc);
    let acc_new = k.op2(OP_F_ADD, f32_, acc_old, term);
    k.b.inst(Section::Code, OP_STORE, &[acc, acc_new]);
    k.branch(inner_cont);

    k.b.inst(Section::Code, OP_LABEL, &[inner_cont]);
    let j_old = k.op1(OP_LOAD, u32_, elem);
    let j_next = k.op2(OP_I_ADD, u32_, j_old, c1);
    k.b.inst(Section::Code, OP_STORE, &[elem, j_next]);
    k.branch(inner_header);

    k.b.inst(Section::Code, OP_LABEL, &[inner_merge]);
    k.branch(outer_cont);

    k.b.inst(Section::Code, OP_LABEL, &[outer_cont]);
    let b_old = k.op1(OP_LOAD, u32_, blk);
    let b_next = k.op2(OP_I_ADD, u32_, b_old, c1);
    k.b.inst(Section::Code, OP_STORE, &[blk, b_next]);
    k.branch(outer_header);

    // --- %outer_merge: store the row -------------------------
    k.b.inst(Section::Code, OP_LABEL, &[outer_merge]);
    let total = k.op1(OP_LOAD, f32_, acc);
    let y_ptr = k.b.typed(
        Section::Code,
        OP_ACCESS_CHAIN,
        k.i.ptr_uniform_f32,
        &[k.i.var_y, c0, row],
    );
    k.b.inst(Section::Code, OP_STORE, &[y_ptr, total]);
    k.branch(exit);

    k.b.inst(Section::Code, OP_LABEL, &[exit]);
    k.b.inst(Section::Code, OP_RETURN, &[]);
    k.b.inst(Section::Code, OP_FUNCTION_END, &[]);
}

fn load_push_constant(k: &mut Kernel, member: u32) -> u32 {
    let index = k.u(member);
    let ptr = k.b.typed(
        Section::Code,
        OP_ACCESS_CHAIN,
        k.i.ptr_pc_u32,
        &[k.i.var_pc, index],
    );
    k.op1(OP_LOAD, k.i.u32_, ptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_has_a_valid_header_and_a_complete_instruction_stream() {
        let words = spirv();
        assert_eq!(words[0], MAGIC);
        assert_eq!(words[1], VERSION_1_0);
        let bound = words[3];
        // Walk the stream; a bad word count leaves a remainder or
        // overruns. Only `OpLabel`'s single operand is checked against
        // the id bound -- most operand words are literals, not ids, and
        // `spirv_val_accepts_the_module` is the real authority here.
        let mut i = 5;
        let mut instructions = 0;
        while i < words.len() {
            let count = (words[i] >> 16) as usize;
            assert!(count > 0, "zero-length instruction at word {i}");
            assert!(i + count <= words.len(), "instruction at {i} overruns");
            if (words[i] & 0xffff) as u16 == OP_LABEL {
                assert!(words[i + 1] < bound, "label id exceeds bound {bound}");
            }
            i += count;
            instructions += 1;
        }
        assert_eq!(i, words.len());
        assert!(instructions > 100, "suspiciously small module");
    }

    #[test]
    fn the_module_is_reproducible() {
        assert_eq!(spirv(), spirv());
    }

    #[test]
    fn every_block_label_is_defined_exactly_once_and_branched_to() {
        let words = spirv();
        let mut labels = Vec::new();
        let mut targets = Vec::new();
        let mut i = 5;
        while i < words.len() {
            let count = (words[i] >> 16) as usize;
            let op = (words[i] & 0xffff) as u16;
            let ops = &words[i + 1..i + count];
            match op {
                OP_LABEL => labels.push(ops[0]),
                OP_BRANCH => targets.push(ops[0]),
                OP_BRANCH_CONDITIONAL => {
                    targets.push(ops[1]);
                    targets.push(ops[2]);
                }
                OP_LOOP_MERGE => {
                    targets.push(ops[0]);
                    targets.push(ops[1]);
                }
                OP_SELECTION_MERGE => targets.push(ops[0]),
                _ => {}
            }
            i += count;
        }
        let mut sorted = labels.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), labels.len(), "a label id is defined twice");
        for t in &targets {
            assert!(labels.contains(t), "branch to undefined label {t}");
        }
        // entry, work, 5 outer, 5 inner, exit
        assert_eq!(labels.len(), 13, "block count changed: {labels:?}");
    }

    #[test]
    fn the_subnormal_scale_constant_is_two_to_the_minus_24() {
        assert_eq!(f32::from_bits(F16_SUBNORMAL_SCALE_BITS), 1.0 / 16_777_216.0);
    }

    /// `spirv-val` is the only external authority on whether a
    /// hand-built module is legal SPIR-V. It ships with `glslang` /
    /// the Vulkan SDK. When it is absent the test says so and passes
    /// rather than silently vanishing.
    #[test]
    fn spirv_val_accepts_the_module() {
        use std::io::Write;
        let Ok(out) = std::process::Command::new("spirv-val")
            .arg("--version")
            .output()
        else {
            eprintln!("spirv-val not on PATH: module NOT externally validated");
            return;
        };
        assert!(out.status.success());
        let dir = std::env::temp_dir().join("frink-vulkan-spirv-val");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q8_0_matvec.spv");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&to_bytes(&spirv())).unwrap();
        drop(f);
        let out = std::process::Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.0")
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "spirv-val rejected the module:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
