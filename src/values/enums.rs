use llvm_sys::core::{LLVMTypeOf, LLVMGetTypeKind};
use llvm_sys::LLVMTypeKind;
use llvm_sys::prelude::LLVMValueRef;

use crate::types::{AnyTypeEnum, BasicTypeEnum};
use crate::values::traits::AsValueRef;
use crate::values::{IntValue, FunctionValue, PointerValue, VectorValue, ArrayValue, StructValue, FloatValue, PhiValue, InstructionValue, MetadataValue};

macro_rules! enum_value_set {
    ($enum_name:ident: $($args:ident),*) => (
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $enum_name {
            $(
                $args($args),
            )*
        }

        impl AsValueRef for $enum_name {
            fn as_value_ref(&self) -> LLVMValueRef {
                match *self {
                    $(
                        $enum_name::$args(ref t) => t.as_value_ref(),
                    )*
                }
            }
        }

        $(
            impl From<$args> for $enum_name {
                fn from(value: $args) -> $enum_name {
                    $enum_name::$args(value)
                }
            }

            impl PartialEq<$args> for $enum_name {
                fn eq(&self, other: &$args) -> bool {
                    self.as_value_ref() == other.as_value_ref()
                }
            }

            impl PartialEq<$enum_name> for $args {
                fn eq(&self, other: &$enum_name) -> bool {
                    self.as_value_ref() == other.as_value_ref()
                }
            }
        )*
    );
}

enum_value_set! {AggregateValueEnum: ArrayValue, StructValue}
enum_value_set! {AnyValueEnum: ArrayValue, IntValue, FloatValue, PhiValue, FunctionValue, PointerValue, StructValue, VectorValue, InstructionValue}
enum_value_set! {BasicValueEnum: ArrayValue, IntValue, FloatValue, PointerValue, StructValue, VectorValue}
enum_value_set! {BasicMetadataValueEnum: ArrayValue, IntValue, FloatValue, PointerValue, StructValue, VectorValue, MetadataValue}

impl AnyValueEnum {
    pub(crate) fn new(value: LLVMValueRef) -> AnyValueEnum {
        let type_kind = unsafe {
            LLVMGetTypeKind(LLVMTypeOf(value))
        };

        match type_kind {
            LLVMTypeKind::LLVMFloatTypeKind |
            LLVMTypeKind::LLVMFP128TypeKind |
            LLVMTypeKind::LLVMDoubleTypeKind |
            LLVMTypeKind::LLVMHalfTypeKind |
            LLVMTypeKind::LLVMX86_FP80TypeKind |
            LLVMTypeKind::LLVMPPC_FP128TypeKind => AnyValueEnum::FloatValue(FloatValue::new(value)),
            LLVMTypeKind::LLVMIntegerTypeKind => AnyValueEnum::IntValue(IntValue::new(value)),
            LLVMTypeKind::LLVMStructTypeKind => AnyValueEnum::StructValue(StructValue::new(value)),
            LLVMTypeKind::LLVMPointerTypeKind => AnyValueEnum::PointerValue(PointerValue::new(value)),
            LLVMTypeKind::LLVMArrayTypeKind => AnyValueEnum::ArrayValue(ArrayValue::new(value)),
            LLVMTypeKind::LLVMVectorTypeKind => AnyValueEnum::VectorValue(VectorValue::new(value)),
            LLVMTypeKind::LLVMFunctionTypeKind => AnyValueEnum::FunctionValue(FunctionValue::new(value).unwrap()),
            LLVMTypeKind::LLVMVoidTypeKind => panic!("Void values shouldn't exist."),
            LLVMTypeKind::LLVMMetadataTypeKind => panic!("Metadata values are not supported as AnyValue's."),
            _ => panic!("The given type is not supported.")
        }
    }

    pub fn get_type(&self) -> AnyTypeEnum {
        let type_ = unsafe {
            LLVMTypeOf(self.as_value_ref())
        };

        AnyTypeEnum::new(type_)
    }
}

impl BasicValueEnum {
    pub(crate) fn new(value: LLVMValueRef) -> BasicValueEnum {
        let type_kind = unsafe {
            LLVMGetTypeKind(LLVMTypeOf(value))
        };

        match type_kind {
            LLVMTypeKind::LLVMFloatTypeKind |
            LLVMTypeKind::LLVMFP128TypeKind |
            LLVMTypeKind::LLVMDoubleTypeKind |
            LLVMTypeKind::LLVMHalfTypeKind |
            LLVMTypeKind::LLVMX86_FP80TypeKind |
            LLVMTypeKind::LLVMPPC_FP128TypeKind => BasicValueEnum::FloatValue(FloatValue::new(value)),
            LLVMTypeKind::LLVMIntegerTypeKind => BasicValueEnum::IntValue(IntValue::new(value)),
            LLVMTypeKind::LLVMStructTypeKind => BasicValueEnum::StructValue(StructValue::new(value)),
            LLVMTypeKind::LLVMPointerTypeKind => BasicValueEnum::PointerValue(PointerValue::new(value)),
            LLVMTypeKind::LLVMArrayTypeKind => BasicValueEnum::ArrayValue(ArrayValue::new(value)),
            LLVMTypeKind::LLVMVectorTypeKind => BasicValueEnum::VectorValue(VectorValue::new(value)),
            _ => unreachable!("The given type is not a basic type."),
        }
    }

    pub fn get_type(&self) -> BasicTypeEnum {
        let type_ = unsafe {
            LLVMTypeOf(self.as_value_ref())
        };

        BasicTypeEnum::new(type_)
    }

    pub fn is_float_value(&self) -> bool {
        match *self {
            BasicValueEnum::FloatValue(_) => true,
            _ => false,
        }
    }

    pub fn is_int_value(&self) -> bool {
        match *self {
            BasicValueEnum::IntValue(_) => true,
            _ => false,
        }
    }

    pub fn is_pointer_value(&self) -> bool {
        match *self {
            BasicValueEnum::PointerValue(_) => true,
            _ => false,
        }
    }

    pub fn into_array_value(self) -> ArrayValue {
        match self {
            BasicValueEnum::ArrayValue(value) => value,
            _ => panic!("not a array value"),
        }
    }

    pub fn into_float_value(self) -> FloatValue {
        match self {
            BasicValueEnum::FloatValue(value) => value,
            _ => panic!("not a float value"),
        }
    }

    pub fn into_int_value(self) -> IntValue {
        match self {
            BasicValueEnum::IntValue(value) => value,
            _ => panic!("not a int value"),
        }
    }

    pub fn into_pointer_value(self) -> PointerValue {
        match self {
            BasicValueEnum::PointerValue(value) => value,
            _ => panic!("not a pointer value"),
        }
    }

    pub fn into_struct_value(self) -> StructValue {
        match self {
            BasicValueEnum::StructValue(value) => value,
            _ => panic!("not a struct value"),
        }
    }

    pub fn into_vector_value(self) -> VectorValue {
        match self {
            BasicValueEnum::VectorValue(value) => value,
            _ => panic!("not a vector value"),
        }
    }
}

impl AggregateValueEnum {
    pub(crate) fn new(value: LLVMValueRef) -> AggregateValueEnum {
        let type_kind = unsafe {
            LLVMGetTypeKind(LLVMTypeOf(value))
        };

        match type_kind {
            LLVMTypeKind::LLVMArrayTypeKind => AggregateValueEnum::ArrayValue(ArrayValue::new(value)),
            LLVMTypeKind::LLVMStructTypeKind => AggregateValueEnum::StructValue(StructValue::new(value)),
            _ => unreachable!("The given type is not an aggregate type."),
        }
    }

    pub fn is_array_value(&self) -> bool {
        match *self {
            AggregateValueEnum::ArrayValue(_) => true,
            _ => false,
        }
    }

    pub fn is_struct_value(&self) -> bool {
        match *self {
            AggregateValueEnum::StructValue(_) => true,
            _ => false,
        }
    }
}

impl BasicMetadataValueEnum {
    pub(crate) fn new(value: LLVMValueRef) -> BasicMetadataValueEnum {
        let type_kind = unsafe {
            LLVMGetTypeKind(LLVMTypeOf(value))
        };

        match type_kind {
            LLVMTypeKind::LLVMFloatTypeKind |
            LLVMTypeKind::LLVMFP128TypeKind |
            LLVMTypeKind::LLVMDoubleTypeKind |
            LLVMTypeKind::LLVMHalfTypeKind |
            LLVMTypeKind::LLVMX86_FP80TypeKind |
            LLVMTypeKind::LLVMPPC_FP128TypeKind => BasicMetadataValueEnum::FloatValue(FloatValue::new(value)),
            LLVMTypeKind::LLVMIntegerTypeKind => BasicMetadataValueEnum::IntValue(IntValue::new(value)),
            LLVMTypeKind::LLVMStructTypeKind => BasicMetadataValueEnum::StructValue(StructValue::new(value)),
            LLVMTypeKind::LLVMPointerTypeKind => BasicMetadataValueEnum::PointerValue(PointerValue::new(value)),
            LLVMTypeKind::LLVMArrayTypeKind => BasicMetadataValueEnum::ArrayValue(ArrayValue::new(value)),
            LLVMTypeKind::LLVMVectorTypeKind => BasicMetadataValueEnum::VectorValue(VectorValue::new(value)),
            LLVMTypeKind::LLVMMetadataTypeKind => BasicMetadataValueEnum::MetadataValue(MetadataValue::new(value)),
            _ => unreachable!("Unsupported type"),
        }
    }

    pub fn as_float_value(&self) -> &FloatValue {
        match *self {
            BasicMetadataValueEnum::FloatValue(ref value) => value,
            _ => panic!("not a float value"),
        }
    }

    pub fn as_int_value(&self) -> &IntValue {
        match *self {
            BasicMetadataValueEnum::IntValue(ref value) => value,
            _ => panic!("not a int value"),
        }
    }

    pub fn as_metadata_value(&self) -> &MetadataValue {
        match *self {
            BasicMetadataValueEnum::MetadataValue(ref value) => value,
            _ => panic!("not a metadata value"),
        }
    }
}

impl From<BasicValueEnum> for AnyValueEnum {
    fn from(value: BasicValueEnum) -> AnyValueEnum {
        AnyValueEnum::new(value.as_value_ref())
    }
}
