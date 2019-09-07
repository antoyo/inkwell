// based on 3fcc5946b219a6b990a372222ea302dd08f6017b

//! Inkwell documentation is a work in progress.
//!
//! If you have any LLVM knowledge that could be used to improve these docs, we would greatly appreciate you opening an issue and/or a pull request on our [GitHub page](https://github.com/TheDan64/inkwell).
//!
//! Due to a rustdoc issue, this documentation represents only the latest supported LLVM version. We hope that this issue will be resolved in the future so that multiple versions can be documented side by side.
//!
//! # Library Wide Notes
//!
//! * Most functions which take a string slice as input may possibly panic in the unlikely event that a c style string cannot be created based on it. (IE if your slice already has a null byte in it)

#![deny(missing_debug_implementations)]
extern crate either;
#[macro_use]
extern crate enum_methods;
extern crate libc;
extern crate llvm_sys;
#[macro_use]
extern crate lazy_static;

#[macro_use]
pub mod support;
#[deny(missing_docs)]
pub mod attributes;
#[deny(missing_docs)]
#[cfg(not(any(feature = "llvm3-6", feature = "llvm3-7", feature = "llvm3-8", feature = "llvm3-9",
              feature = "llvm4-0", feature = "llvm5-0", feature = "llvm6-0")))]
pub mod comdat;
#[deny(missing_docs)]
pub mod basic_block;
pub mod builder;
#[deny(missing_docs)]
pub mod context;
pub mod data_layout;
pub mod execution_engine;
pub mod memory_buffer;
#[deny(missing_docs)]
pub mod module;
pub mod object_file;
pub mod passes;
pub mod targets;
pub mod types;
pub mod values;

use llvm_sys::{LLVMIntPredicate, LLVMRealPredicate, LLVMVisibility, LLVMThreadLocalMode, LLVMDLLStorageClass, LLVMAtomicOrdering, LLVMAtomicRMWBinOp};

use std::convert::TryFrom;

// Thanks to kennytm for coming up with assert_unique_features!
// which ensures that the LLVM feature flags are mutually exclusive
macro_rules! assert_unique_features {
    () => {};
    ($first:tt $(,$rest:tt)*) => {
        $(
            #[cfg(all(feature = $first, feature = $rest))]
            compile_error!(concat!("features \"", $first, "\" and \"", $rest, "\" cannot be used together"));
        )*
        assert_unique_features!($($rest),*);
    }
}

// This macro ensures that at least one of the LLVM feature
// flags are provided and prints them out if none are provided
macro_rules! assert_used_features {
    ($($all:tt),*) => {
        #[cfg(not(any($(feature = $all),*)))]
        compile_error!(concat!("One of the LLVM feature flags must be provided: ", $($all, " "),*));
    }
}

macro_rules! assert_unique_used_features {
    ($($all:tt),*) => {
        assert_unique_features!($($all),*);
        assert_used_features!($($all),*);
    }
}

assert_unique_used_features!{"llvm3-6", "llvm3-7", "llvm3-8", "llvm3-9", "llvm4-0", "llvm5-0", "llvm6-0", "llvm7-0", "llvm8-0"}

/// Defines the address space in which a global will be inserted.
///
/// # Remarks
/// See also: https://llvm.org/doxygen/NVPTXBaseInfo_8h_source.html
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum AddressSpace {
    Generic = 0,
    Global  = 1,
    Shared  = 3,
    Const   = 4,
    Local   = 5,
}

impl TryFrom<u32> for AddressSpace {
    type Error = ();

    fn try_from(val: u32) -> Result<Self, Self::Error> {
        match val {
            0 => Ok(AddressSpace::Generic),
            1 => Ok(AddressSpace::Global),
            3 => Ok(AddressSpace::Shared),
            4 => Ok(AddressSpace::Const),
            5 => Ok(AddressSpace::Local),
            _ => Err(()),
        }
    }
}

// REVIEW: Maybe this belongs in some sort of prelude?
/// This enum defines how to compare a `left` and `right` `IntValue`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IntPredicate {
    /// Equal
    EQ,

    /// Not Equal
    NE,

    /// Unsigned Greater Than
    UGT,

    /// Unsigned Greater Than or Equal
    UGE,

    /// Unsigned Less Than
    ULT,

    /// Unsigned Less Than or Equal
    ULE,

    /// Signed Greater Than
    SGT,

    /// Signed Greater Than or Equal
    SGE,

    /// Signed Less Than
    SLT,

    /// Signed Less Than or Equal
    SLE,
}

impl IntPredicate {
    pub fn new(int_pred: LLVMIntPredicate) -> Self {
        int_pred.into()
    }
}

impl From<IntPredicate> for LLVMIntPredicate {
    fn from(int_pred: IntPredicate) -> Self {
        match int_pred {
            IntPredicate::EQ => LLVMIntPredicate::LLVMIntEQ,
            IntPredicate::NE => LLVMIntPredicate::LLVMIntNE,
            IntPredicate::UGT => LLVMIntPredicate::LLVMIntUGT,
            IntPredicate::UGE => LLVMIntPredicate::LLVMIntUGE,
            IntPredicate::ULT => LLVMIntPredicate::LLVMIntULT,
            IntPredicate::ULE => LLVMIntPredicate::LLVMIntULE,
            IntPredicate::SGT => LLVMIntPredicate::LLVMIntSGT,
            IntPredicate::SGE => LLVMIntPredicate::LLVMIntSGE,
            IntPredicate::SLT => LLVMIntPredicate::LLVMIntSLT,
            IntPredicate::SLE => LLVMIntPredicate::LLVMIntSLE,
        }
    }
}

impl From<LLVMIntPredicate> for IntPredicate {
    fn from(int_pred: LLVMIntPredicate) -> Self {
        match int_pred {
            LLVMIntPredicate::LLVMIntEQ => IntPredicate::EQ,
            LLVMIntPredicate::LLVMIntNE => IntPredicate::NE,
            LLVMIntPredicate::LLVMIntUGT => IntPredicate::UGT,
            LLVMIntPredicate::LLVMIntUGE => IntPredicate::UGE,
            LLVMIntPredicate::LLVMIntULT => IntPredicate::ULT,
            LLVMIntPredicate::LLVMIntULE => IntPredicate::ULE,
            LLVMIntPredicate::LLVMIntSGT => IntPredicate::SGT,
            LLVMIntPredicate::LLVMIntSGE => IntPredicate::SGE,
            LLVMIntPredicate::LLVMIntSLT => IntPredicate::SLT,
            LLVMIntPredicate::LLVMIntSLE => IntPredicate::SLE,
        }
    }
}

// REVIEW: Maybe this belongs in some sort of prelude?
/// Defines how to compare a `left` and `right` `FloatValue`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FloatPredicate {
    /// Returns true if `left` == `right` and neither are NaN
    OEQ,

    /// Returns true if `left` >= `right` and neither are NaN
    OGE,

    /// Returns true if `left` > `right` and neither are NaN
    OGT,

    /// Returns true if `left` <= `right` and neither are NaN
    OLE,

    /// Returns true if `left` < `right` and neither are NaN
    OLT,

    /// Returns true if `left` != `right` and neither are NaN
    ONE,

    /// Returns true if neither value is NaN
    ORD,

    /// Always returns false
    PredicateFalse,

    /// Always returns true
    PredicateTrue,

    /// Returns true if `left` == `right` or either is NaN
    UEQ,

    /// Returns true if `left` >= `right` or either is NaN
    UGE,

    /// Returns true if `left` > `right` or either is NaN
    UGT,

    /// Returns true if `left` <= `right` or either is NaN
    ULE,

    /// Returns true if `left` < `right` or either is NaN
    ULT,

    /// Returns true if `left` != `right` or either is NaN
    UNE,

    /// Returns true if either value is NaN
    UNO,
}

impl FloatPredicate {
    pub fn new(float_pred: LLVMRealPredicate) -> Self {
        float_pred.into()
    }
}

impl From<FloatPredicate> for LLVMRealPredicate {
    fn from(float_pred: FloatPredicate) -> Self {
        match float_pred {
            FloatPredicate::OEQ => LLVMRealPredicate::LLVMRealOEQ,
            FloatPredicate::OGE => LLVMRealPredicate::LLVMRealOGE,
            FloatPredicate::OGT => LLVMRealPredicate::LLVMRealOGT,
            FloatPredicate::OLE => LLVMRealPredicate::LLVMRealOLE,
            FloatPredicate::OLT => LLVMRealPredicate::LLVMRealOLT,
            FloatPredicate::ONE => LLVMRealPredicate::LLVMRealONE,
            FloatPredicate::ORD => LLVMRealPredicate::LLVMRealORD,
            FloatPredicate::PredicateFalse => LLVMRealPredicate::LLVMRealPredicateFalse,
            FloatPredicate::PredicateTrue => LLVMRealPredicate::LLVMRealPredicateTrue,
            FloatPredicate::UEQ => LLVMRealPredicate::LLVMRealUEQ,
            FloatPredicate::UGE => LLVMRealPredicate::LLVMRealUGE,
            FloatPredicate::UGT => LLVMRealPredicate::LLVMRealUGT,
            FloatPredicate::ULE => LLVMRealPredicate::LLVMRealULE,
            FloatPredicate::ULT => LLVMRealPredicate::LLVMRealULT,
            FloatPredicate::UNE => LLVMRealPredicate::LLVMRealUNE,
            FloatPredicate::UNO => LLVMRealPredicate::LLVMRealUNO,
        }
    }
}

impl From<LLVMRealPredicate> for FloatPredicate {
    fn from(float_pred: LLVMRealPredicate) -> Self {
        match float_pred {
            LLVMRealPredicate::LLVMRealOEQ => FloatPredicate::OEQ,
            LLVMRealPredicate::LLVMRealOGE => FloatPredicate::OGE,
            LLVMRealPredicate::LLVMRealOGT => FloatPredicate::OGT,
            LLVMRealPredicate::LLVMRealOLE => FloatPredicate::OLE,
            LLVMRealPredicate::LLVMRealOLT => FloatPredicate::OLT,
            LLVMRealPredicate::LLVMRealONE => FloatPredicate::ONE,
            LLVMRealPredicate::LLVMRealORD => FloatPredicate::ORD,
            LLVMRealPredicate::LLVMRealPredicateFalse => FloatPredicate::PredicateFalse,
            LLVMRealPredicate::LLVMRealPredicateTrue => FloatPredicate::PredicateTrue,
            LLVMRealPredicate::LLVMRealUEQ => FloatPredicate::UEQ,
            LLVMRealPredicate::LLVMRealUGE => FloatPredicate::UGE,
            LLVMRealPredicate::LLVMRealUGT => FloatPredicate::UGT,
            LLVMRealPredicate::LLVMRealULE => FloatPredicate::ULE,
            LLVMRealPredicate::LLVMRealULT => FloatPredicate::ULT,
            LLVMRealPredicate::LLVMRealUNE => FloatPredicate::UNE,
            LLVMRealPredicate::LLVMRealUNO => FloatPredicate::UNO,
        }
    }
}

// REVIEW: Maybe this belongs in some sort of prelude?
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AtomicOrdering {
    NotAtomic,
    Unordered,
    Monotonic,
    Acquire,
    Release,
    AcquireRelease,
    SequentiallyConsistent,
}

impl From<AtomicOrdering> for LLVMAtomicOrdering {
    fn from(ordering: AtomicOrdering) -> Self {
        match ordering {
            AtomicOrdering::NotAtomic => LLVMAtomicOrdering::LLVMAtomicOrderingNotAtomic,
            AtomicOrdering::Unordered => LLVMAtomicOrdering::LLVMAtomicOrderingUnordered,
            AtomicOrdering::Monotonic => LLVMAtomicOrdering::LLVMAtomicOrderingMonotonic,
            AtomicOrdering::Acquire => LLVMAtomicOrdering::LLVMAtomicOrderingAcquire,
            AtomicOrdering::Release => LLVMAtomicOrdering::LLVMAtomicOrderingRelease,
            AtomicOrdering::AcquireRelease => LLVMAtomicOrdering::LLVMAtomicOrderingAcquireRelease,
            AtomicOrdering::SequentiallyConsistent => LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AtomicRMWBinOp {
    /// Stores to memory and returns the prior value.
    Xchg,

    /// Adds to the value in memory and returns the prior value.
    Add,

    /// Subtract a value off the value in memory and returns the prior value.
    Sub,

    /// Bitwise and into memory and returns the prior value.
    And,

    /// Bitwise nands into memory and returns the prior value.
    Nand,

    /// Bitwise ors into memory and returns the prior value.
    Or,

    /// Bitwise xors into memory and returns the prior value.
    Xor,

    /// Sets memory to the signed-greater of the value provided and the value in memory. Returns the value that was in memory.
    Max,

    /// Sets memory to the signed-lesser of the value provided and the value in memory. Returns the value that was in memory.
    Min,

    /// Sets memory to the unsigned-greater of the value provided and the value in memory. Returns the value that was in memory.
    UMax,

    /// Sets memory to the unsigned-lesser of the value provided and the value in memory. Returns the value that was in memory.
    UMin,
}

impl From<AtomicRMWBinOp> for LLVMAtomicRMWBinOp {
    fn from(bin_op: AtomicRMWBinOp) -> Self {
        match bin_op {
            AtomicRMWBinOp::Xchg => LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpXchg,
            AtomicRMWBinOp::Add => LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd,
            AtomicRMWBinOp::Sub => LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpSub,
            AtomicRMWBinOp::And => LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAnd,
            AtomicRMWBinOp::Nand => LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpNand,
            AtomicRMWBinOp::Or => LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpOr,
            AtomicRMWBinOp::Xor => LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpXor,
            AtomicRMWBinOp::Max => LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpMax,
            AtomicRMWBinOp::Min => LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpMin,
            AtomicRMWBinOp::UMax => LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpUMax,
            AtomicRMWBinOp::UMin => LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpUMin,
        }
    }
}

/// Defines the optimization level used to compile a `Module`.
///
/// # Remarks
/// See also: https://llvm.org/doxygen/CodeGen_8h_source.html
#[repr(u32)]
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum OptimizationLevel {
    None       = 0,
    Less       = 1,
    Default    = 2,
    Aggressive = 3
}

impl Default for OptimizationLevel {
    /// Returns the default value for `OptimizationLevel`, namely `OptimizationLevel::Default`.
    fn default() -> Self {
        OptimizationLevel::Default
    }
}

enum_rename!{
    GlobalVisibility <=> LLVMVisibility {
        Default <=> LLVMDefaultVisibility,
        Hidden <=> LLVMHiddenVisibility,
        Protected <=> LLVMProtectedVisibility,
    }
}

impl Default for GlobalVisibility {
    /// Returns the default value for `GlobalVisibility`, namely `GlobalVisibility::Default`.
    fn default() -> Self {
        GlobalVisibility::Default
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThreadLocalMode {
    GeneralDynamicTLSModel,
    LocalDynamicTLSModel,
    InitialExecTLSModel,
    LocalExecTLSModel,
}

impl ThreadLocalMode {
    pub(crate) fn new(thread_local_mode: LLVMThreadLocalMode) -> Option<Self> {
        match thread_local_mode {
            LLVMThreadLocalMode::LLVMGeneralDynamicTLSModel => Some(ThreadLocalMode::GeneralDynamicTLSModel),
            LLVMThreadLocalMode::LLVMLocalDynamicTLSModel => Some(ThreadLocalMode::LocalDynamicTLSModel),
            LLVMThreadLocalMode::LLVMInitialExecTLSModel => Some(ThreadLocalMode::InitialExecTLSModel),
            LLVMThreadLocalMode::LLVMLocalExecTLSModel => Some(ThreadLocalMode::LocalExecTLSModel),
            LLVMThreadLocalMode::LLVMNotThreadLocal => None
        }
    }

    pub(crate) fn as_llvm_mode(&self) -> LLVMThreadLocalMode {
        match *self {
            ThreadLocalMode::GeneralDynamicTLSModel => LLVMThreadLocalMode::LLVMGeneralDynamicTLSModel,
            ThreadLocalMode::LocalDynamicTLSModel => LLVMThreadLocalMode::LLVMLocalDynamicTLSModel,
            ThreadLocalMode::InitialExecTLSModel => LLVMThreadLocalMode::LLVMInitialExecTLSModel,
            ThreadLocalMode::LocalExecTLSModel => LLVMThreadLocalMode::LLVMLocalExecTLSModel,
            // None => LLVMThreadLocalMode::LLVMNotThreadLocal,
        }
    }
}

enum_rename! {
    DLLStorageClass <=> LLVMDLLStorageClass {
        Default <=> LLVMDefaultStorageClass,
        Import <=> LLVMDLLImportStorageClass,
        Export <=> LLVMDLLExportStorageClass,
    }
}

impl Default for DLLStorageClass {
    /// Returns the default value for `DLLStorageClass`, namely `DLLStorageClass::Default`.
    fn default() -> Self {
        DLLStorageClass::Default
    }
}
