from xdsl.dialects.builtin import (
    Float32Type,
    IndexType,
    MemRefType,
)
from xdsl.ir import Dialect, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)


# Since we need tp define rvv type
# like   vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
@irdl_attr_definition
class RVVvectorType(ParametrizedAttribute):
    # class RVVVectorType(ParametrizedAttribute, TypeAttribute):

    """RVV custom type - represents vfloat32m1_t"""

    name = "rvv.vec"


@irdl_attr_definition
# class RVVFloat32M1Type(ParametrizedAttribute):
# class RVVFloat32M1Type(ParametrizedAttribute):
class RVVFloat32M1Type(ParametrizedAttribute, TypeAttribute):
    name = "rvv.vfloat32m1"


@irdl_attr_definition
class RVVFloat32M2Type(ParametrizedAttribute):
    """Represents vfloat32m2_t"""

    name = "rvv.vfloat32m2"


@irdl_op_definition
class SetvlOp(IRDLOperation):
    """
    Set vector length
    """

    name = "rvv.setvl"

    # Application vector length, we pass on the size of the index
    avl = operand_def(IndexType)
    vl = result_def(IndexType)

    def __init__(self, avl: SSAValue):
        super().__init__(
            operands=[avl],
            result_types=[IndexType()],
        )


@irdl_op_definition
class vle32_v_f32m1Op(IRDLOperation):
    """Vector load operation"""

    name = "rvv.vle32_v_f32m1Op"

    memref = operand_def(MemRefType)
    offset = operand_def(IndexType)
    avl = operand_def(IndexType)
    # vl = result_def(RVVvectorType)
    vl = result_def(RVVFloat32M1Type)

    def __init__(self, memref: SSAValue, offset: SSAValue, avl: SSAValue):
        super().__init__(
            operands=[memref, offset, avl],
            # result_types=[RVVFloat32M1Type([])],  # No parameters!
            result_types=[RVVFloat32M1Type()],  # No parameters!
            # result_types=[RVVvectorType([Float32Type(), IntAttr(1)])],  # Use list constructor
        )


@irdl_op_definition
class vfmacc_vf_f32m1Op(IRDLOperation):
    """
    Fused multiply-accumulate: vd = vd + memref[offset] * vs
    C_reg_0_0 = __riscv_vfmacc_vf_f32m1(C_reg_0_0, B[(k) * (ldb)], A_reg_0, (vl));

    """

    name = "rvv.vfmacc_vf_f32m1Op"

    vd = operand_def(RVVFloat32M1Type)
    memref = operand_def(MemRefType)
    offset = operand_def(IndexType)
    vs = operand_def(RVVFloat32M1Type)
    avl = operand_def(IndexType)

    result = result_def(RVVFloat32M1Type)

    def __init__(
        self,
        vd: SSAValue,
        memref: SSAValue,
        offset: SSAValue,
        vs: SSAValue,
        avl: SSAValue,
    ):
        super().__init__(
            operands=[vd, memref, offset, vs, avl],
            result_types=[RVVFloat32M1Type()],  # No parameters!
        )


@irdl_op_definition
class vfmv_v_f_f32m1Op(IRDLOperation):
    """
    C_reg_0_0 = __riscv_vfmv_v_f_f32m1(0.0f, (vl));
    """

    name = "rvv.vfmv_v_f_f32m1"

    scalar = operand_def(Float32Type)
    avl = operand_def(IndexType)

    result = result_def(RVVFloat32M1Type)

    def __init__(self, scalar: SSAValue, avl: SSAValue):
        super().__init__(operands=[scalar, avl], result_types=[RVVFloat32M1Type()])


@irdl_op_definition
class vse32_v_f32m1Op(IRDLOperation):
    """
    Implement    __riscv_vse32_v_f32m1(&C[0], C_reg_0_0, (vl));
    """

    name = "rvv.vse32_v_f32m1Op"

    memref = operand_def(MemRefType)
    offset = operand_def(IndexType)
    vec = operand_def(RVVFloat32M1Type)
    avl = operand_def(IndexType)

    def __init__(
        self, memref: SSAValue, offset: SSAValue, vector: SSAValue, avl: SSAValue
    ):
        super().__init__(
            operands=[memref, offset, vector, avl],
            result_types=[],  # Empty, this func dont return anything
        )


RVV = Dialect(
    "rvv",
    [
        SetvlOp,
        vle32_v_f32m1Op,
        vfmacc_vf_f32m1Op,
        vfmv_v_f_f32m1Op,
        vse32_v_f32m1Op,
    ],
    [
        RVVvectorType,
        RVVFloat32M1Type,
    ],
)
