use llvm_sys::core::LLVMGetTypeKind;
use llvm_sys::LLVMTypeKind;
use llvm_sys::prelude::LLVMTypeRef;

use crate::types::{IntType, VoidType, FunctionType, PointerType, VectorType, ArrayType, StructType, FloatType};
use crate::types::traits::AsTypeRef;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum AnyTypeEnum {
    /// A contiguous homogeneous container type.
    ArrayType(ArrayType),
    /// A floating point type.
    FloatType(FloatType),
    /// A function return and parameter definition.
    FunctionType(FunctionType),
    /// An integer type.
    IntType(IntType),
    /// A pointer type.
    PointerType(PointerType),
    /// A contiguous heterogeneous container type.
    StructType(StructType),
    /// A contiguous homogeneous "SIMD" container type.
    VectorType(VectorType),
    /// A valueless type.
    VoidType(VoidType),
}

impl AsTypeRef for AnyTypeEnum {
    fn as_type_ref(&self) -> LLVMTypeRef {
        match *self {
            AnyTypeEnum::ArrayType(ref t) => t.as_type_ref(),
            AnyTypeEnum::FloatType(ref t) => t.as_type_ref(),
            AnyTypeEnum::FunctionType(ref t) => t.as_type_ref(),
            AnyTypeEnum::IntType(ref t) => t.as_type_ref(),
            AnyTypeEnum::PointerType(ref t) => t.as_type_ref(),
            AnyTypeEnum::StructType(ref t) => t.as_type_ref(),
            AnyTypeEnum::VectorType(ref t) => t.as_type_ref(),
            AnyTypeEnum::VoidType(ref t) => t.as_type_ref(),
        }
    }
}

impl From<IntType> for AnyTypeEnum {
    fn from(value: IntType) -> AnyTypeEnum {
        AnyTypeEnum::IntType(value)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum BasicTypeEnum {
    /// A contiguous homogeneous container type.
    ArrayType(ArrayType),
    /// A floating point type.
    FloatType(FloatType),
    // An integer type.
    IntType(IntType),
    /// A pointer type.
    PointerType(PointerType),
    /// A contiguous heterogeneous container type.
    StructType(StructType),
    /// A contiguous homogeneous "SIMD" container type.
    VectorType(VectorType),
}

impl AsTypeRef for BasicTypeEnum {
    fn as_type_ref(&self) -> LLVMTypeRef {
        match *self {
            BasicTypeEnum::ArrayType(ref t) => t.as_type_ref(),
            BasicTypeEnum::FloatType(ref t) => t.as_type_ref(),
            BasicTypeEnum::IntType(ref t) => t.as_type_ref(),
            BasicTypeEnum::PointerType(ref t) => t.as_type_ref(),
            BasicTypeEnum::StructType(ref t) => t.as_type_ref(),
            BasicTypeEnum::VectorType(ref t) => t.as_type_ref(),
        }
    }
}

impl AnyTypeEnum {
    pub(crate) fn new(type_: LLVMTypeRef) -> AnyTypeEnum {
        let type_kind = unsafe {
            LLVMGetTypeKind(type_)
        };

        match type_kind {
            LLVMTypeKind::LLVMVoidTypeKind => AnyTypeEnum::VoidType(VoidType::new(type_)),
            LLVMTypeKind::LLVMHalfTypeKind |
            LLVMTypeKind::LLVMFloatTypeKind |
            LLVMTypeKind::LLVMDoubleTypeKind |
            LLVMTypeKind::LLVMX86_FP80TypeKind |
            LLVMTypeKind::LLVMFP128TypeKind |
            LLVMTypeKind::LLVMPPC_FP128TypeKind => AnyTypeEnum::FloatType(FloatType::new(type_)),
            LLVMTypeKind::LLVMLabelTypeKind => panic!("FIXME: Unsupported type: Label"),
            LLVMTypeKind::LLVMIntegerTypeKind => AnyTypeEnum::IntType(IntType::new(type_)),
            LLVMTypeKind::LLVMFunctionTypeKind => AnyTypeEnum::FunctionType(FunctionType::new(type_)),
            LLVMTypeKind::LLVMStructTypeKind => AnyTypeEnum::StructType(StructType::new(type_)),
            LLVMTypeKind::LLVMArrayTypeKind => AnyTypeEnum::ArrayType(ArrayType::new(type_)),
            LLVMTypeKind::LLVMPointerTypeKind => AnyTypeEnum::PointerType(PointerType::new(type_)),
            LLVMTypeKind::LLVMVectorTypeKind => AnyTypeEnum::VectorType(VectorType::new(type_)),
            LLVMTypeKind::LLVMMetadataTypeKind => panic!("FIXME: Unsupported type: Metadata"),
            LLVMTypeKind::LLVMX86_MMXTypeKind => panic!("FIXME: Unsupported type: MMX"),
            #[cfg(not(any(feature = "llvm3-6", feature = "llvm3-7")))]
            LLVMTypeKind::LLVMTokenTypeKind => panic!("FIXME: Unsupported type: Token"),
        }
    }

    pub fn into_array_type(self) -> ArrayType {
        match self {
            AnyTypeEnum::ArrayType(typ) => typ,
            _ => panic!("enum not a array type"),
        }
    }

    pub fn into_function_type(self) -> FunctionType {
        match self {
            AnyTypeEnum::FunctionType(typ) => typ,
            _ => panic!("enum not a function type"),
        }
    }

    pub fn into_float_type(self) -> FloatType {
        match self {
            AnyTypeEnum::FloatType(typ) => typ,
            _ => panic!("enum not a float type"),
        }
    }

    pub fn into_int_type(self) -> IntType {
        match self {
            AnyTypeEnum::IntType(typ) => typ,
            _ => panic!("enum not a int type"),
        }
    }

    pub fn into_pointer_type(self) -> PointerType {
        match self {
            AnyTypeEnum::PointerType(typ) => typ,
            _ => panic!("enum not a pointer type"),
        }
    }

    pub fn into_struct_type(self) -> StructType {
        match self {
            AnyTypeEnum::StructType(typ) => typ,
            _ => panic!("enum not a struct type"),
        }
    }

    pub fn into_vector_type(self) -> VectorType {
        match self {
            AnyTypeEnum::VectorType(typ) => typ,
            _ => panic!("enum not a vector type"),
        }
    }

    /// This will panic if type is a void or function type.
    pub(crate) fn to_basic_type_enum(&self) -> BasicTypeEnum {
        BasicTypeEnum::new(self.as_type_ref())
    }
}

impl BasicTypeEnum {
    pub(crate) fn new(type_: LLVMTypeRef) -> BasicTypeEnum {
        let type_kind = unsafe {
            LLVMGetTypeKind(type_)
        };

        match type_kind {
            LLVMTypeKind::LLVMHalfTypeKind |
            LLVMTypeKind::LLVMFloatTypeKind |
            LLVMTypeKind::LLVMDoubleTypeKind |
            LLVMTypeKind::LLVMX86_FP80TypeKind |
            LLVMTypeKind::LLVMFP128TypeKind |
            LLVMTypeKind::LLVMPPC_FP128TypeKind => BasicTypeEnum::FloatType(FloatType::new(type_)),
            LLVMTypeKind::LLVMIntegerTypeKind => BasicTypeEnum::IntType(IntType::new(type_)),
            LLVMTypeKind::LLVMStructTypeKind => BasicTypeEnum::StructType(StructType::new(type_)),
            LLVMTypeKind::LLVMPointerTypeKind => BasicTypeEnum::PointerType(PointerType::new(type_)),
            LLVMTypeKind::LLVMArrayTypeKind => BasicTypeEnum::ArrayType(ArrayType::new(type_)),
            LLVMTypeKind::LLVMVectorTypeKind => BasicTypeEnum::VectorType(VectorType::new(type_)),
            LLVMTypeKind::LLVMMetadataTypeKind => unreachable!("Unsupported basic type: Metadata"),
            LLVMTypeKind::LLVMX86_MMXTypeKind => unreachable!("Unsupported basic type: MMX"),
            LLVMTypeKind::LLVMLabelTypeKind => unreachable!("Unsupported basic type: Label"),
            LLVMTypeKind::LLVMVoidTypeKind => unreachable!("Unsupported basic type: VoidType"),
            LLVMTypeKind::LLVMFunctionTypeKind => unreachable!("Unsupported basic type: FunctionType"),
            #[cfg(not(any(feature = "llvm3-6", feature = "llvm3-7")))]
            LLVMTypeKind::LLVMTokenTypeKind => unreachable!("Unsupported basic type: Token"),
        }
    }

    pub fn as_float_type(&self) -> &FloatType {
        match *self {
            BasicTypeEnum::FloatType(ref typ) => typ,
            _ => panic!("not a float type"),
        }
    }

    pub fn as_int_type(&self) -> &IntType {
        match *self {
            BasicTypeEnum::IntType(ref typ) => typ,
            _ => panic!("not a int type"),
        }
    }

    pub fn is_array_type(&self) -> bool {
        match *self {
            BasicTypeEnum::ArrayType(_) => true,
            _ => false,
        }
    }

    pub fn is_vector_type(&self) -> bool {
        match *self {
            BasicTypeEnum::VectorType(_) => true,
            _ => false,
        }
    }

    pub fn into_array_type(self) -> ArrayType {
        match self {
            BasicTypeEnum::ArrayType(typ) => typ,
            _ => panic!("not a array type"),
        }
    }

    pub fn into_float_type(self) -> FloatType {
        match self {
            BasicTypeEnum::FloatType(typ) => typ,
            _ => panic!("not a float type"),
        }
    }

    pub fn into_int_type(self) -> IntType {
        match self {
            BasicTypeEnum::IntType(typ) => typ,
            _ => panic!("not a int type"),
        }
    }

    pub fn into_pointer_type(self) -> PointerType {
        match self {
            BasicTypeEnum::PointerType(typ) => typ,
            _ => panic!("not a pointer type"),
        }
    }

    pub fn into_struct_type(self) -> StructType {
        match self {
            BasicTypeEnum::StructType(typ) => typ,
            _ => panic!("not a struct type"),
        }
    }

    pub fn into_vector_type(self) -> VectorType {
        match self {
            BasicTypeEnum::VectorType(typ) => typ,
            _ => panic!("not a vector type"),
        }
    }
}

impl From<StructType> for BasicTypeEnum {
    fn from(typ: StructType) -> Self {
        BasicTypeEnum::StructType(typ)
    }
}

impl From<ArrayType> for BasicTypeEnum {
    fn from(typ: ArrayType) -> Self {
        BasicTypeEnum::ArrayType(typ)
    }
}

impl From<PointerType> for BasicTypeEnum {
    fn from(typ: PointerType) -> Self {
        BasicTypeEnum::PointerType(typ)
    }
}

impl From<FloatType> for BasicTypeEnum {
    fn from(typ: FloatType) -> Self {
        BasicTypeEnum::FloatType(typ)
    }
}

impl From<IntType> for BasicTypeEnum {
    fn from(typ: IntType) -> Self {
        BasicTypeEnum::IntType(typ)
    }
}

impl From<VectorType> for BasicTypeEnum {
    fn from(typ: VectorType) -> Self {
        BasicTypeEnum::VectorType(typ)
    }
}
