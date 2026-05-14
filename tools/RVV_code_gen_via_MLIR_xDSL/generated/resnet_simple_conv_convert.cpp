// Auto-generated C code for conv-style linalg graph
// Generated from MLIR->emitC->C pipeline (conv-aware backend)
#include <stdint.h>
#include <string.h>
#include <datatypes.h>
#include <npu_highlevel.h>
#include <primitive.h>

extern int conv_operator(Tensor *tensor_in, Tensor *tensor_out, Tensor *tensor_orig, CONV_OPTION *conv_option);

#define BASE_SCRATCHPAD0 0x90000000u
#define BASE_SCRATCHPAD1 0x90020000u
#define BASE_SCRATCHPAD2 0x90040000u
#define BASE_SCRATCHPAD3 0x90060000u
#define BASE_CIM0 0x00080000u
#define CIM_PAGE_BYTES 0x00002000u
#define BASE_CIM_FC (BASE_CIM0 + 2u * CIM_PAGE_BYTES)

static inline void make_tensor(Tensor *t, uint32_t base_addr, int dim0, int dim1, int dim2, int type_data, int wd_data) {
    int min_stride = min_stride1(dim0, wd_data);
    t->base_addr = base_addr;
    t->dim0 = dim0;
    t->dim1 = dim1;
    t->dim2 = dim2;
    t->type_data = type_data;
    t->wd_data = wd_data;
    t->byte_stride1 = min_stride;
    t->byte_stride2 = min_stride * dim1;
}

void forward_conv_convert(void) {
    npu_mem_init();
    const int td = TYPE_INT;
    const int wd = WIDTH_8;
    Tensor input_host, padded_in, conv_out, bn_tmp, relu_out;
    Tensor pooled_sum, pooled_avg, collapsed, logits, logits_bias;
    Tensor conv_w, fc_w, fc_w_t, fc_w_t_cim;
    Tensor bn_mean, bn_scale, bn_beta, fc_bias;

    make_tensor(&input_host, BASE_SCRATCHPAD3, 3, 4, 4, td, wd);
    make_tensor(&padded_in, BASE_SCRATCHPAD0, 3, 10, 10, td, wd);
    make_tensor(&conv_out, BASE_SCRATCHPAD1, 4, 8, 8, td, wd);
    make_tensor(&bn_tmp, BASE_SCRATCHPAD0, 4, 8, 8, td, wd);
    make_tensor(&relu_out, BASE_SCRATCHPAD1, 4, 8, 8, td, wd);
    make_tensor(&pooled_sum, BASE_SCRATCHPAD3, 4, 1, 1, td, wd);
    make_tensor(&pooled_avg, BASE_SCRATCHPAD0, 4, 1, 1, td, wd);
    make_tensor(&collapsed, BASE_SCRATCHPAD0, 4, 1, 1, td, wd);
    make_tensor(&logits, BASE_SCRATCHPAD0, 2, 1, 1, td, wd);
    make_tensor(&logits_bias, BASE_SCRATCHPAD1, 2, 1, 1, td, wd);
    make_tensor(&conv_w, BASE_CIM0, 4, 3, 9, td, wd);
    make_tensor(&fc_w, BASE_SCRATCHPAD2, 2, 4, 1, td, wd);
    make_tensor(&fc_w_t, BASE_SCRATCHPAD1, 4, 2, 1, td, wd);
    make_tensor(&fc_w_t_cim, BASE_CIM_FC, 4, 2, 1, td, wd);
    make_tensor(&bn_mean, BASE_SCRATCHPAD3, 4, 1, 1, td, wd);
    make_tensor(&bn_scale, BASE_SCRATCHPAD3, 4, 1, 1, td, wd);
    make_tensor(&bn_beta, BASE_SCRATCHPAD3, 4, 1, 1, td, wd);
    make_tensor(&fc_bias, BASE_SCRATCHPAD3, 2, 1, 1, td, wd);

    constantofshape_operator(&padded_in, 0u);
    constantofshape_operator(&conv_out, 0u);
    constantofshape_operator(&bn_tmp, 0u);
    constantofshape_operator(&relu_out, 0u);
    constantofshape_operator(&pooled_sum, 0u);
    constantofshape_operator(&pooled_avg, 0u);
    constantofshape_operator(&logits, 0u);
    constantofshape_operator(&logits_bias, 0u);
    tensor_tensor_add(&input_host, &padded_in, &padded_in);

    for (int kx0 = 0, part = 0; kx0 < 3; kx0 += 2, ++part) {
        int cur_kx = (kx0 + 2 <= 3) ? 2 : (3 - kx0);
        Tensor shifted = padded_in;
        shifted.dim1 = conv_out.dim1 + cur_kx - 1;
        shifted.dim2 = conv_out.dim2 + 3 - 1;
        shifted.base_addr = padded_in.base_addr + (uint32_t)kx0 * (uint32_t)padded_in.byte_stride1;
        CONV_OPTION conv_opt;
        memset(&conv_opt, 0, sizeof(conv_opt));
        conv_opt.type_data = td;
        conv_opt.wd_data = wd;
        conv_opt.byte_base_wt = BASE_CIM0 + (uint32_t)part * CIM_PAGE_BYTES;
        conv_opt.accumulate = (part == 0) ? 0u : 1u;
        conv_opt.size_x = (uint32_t)cur_kx;
        conv_opt.size_y = 3u;
        conv_opt.slide_x = 1u; conv_opt.slide_y = 1u;
        conv_opt.dilate_x = 1u; conv_opt.dilate_y = 1u;
        conv_opt.log2trs_x = 0u; conv_opt.log2trs_y = 0u;
        conv_opt.padding_w = 0u; conv_opt.padding_n = 0u; conv_opt.padding_value = 0u;
        conv_operator(&shifted, &conv_out, &conv_out, &conv_opt);
    }
    tensor_vector_operator(&conv_out, &bn_mean, &bn_tmp, OPERATION_SUB);
    tensor_vector_operator(&bn_tmp, &bn_scale, &bn_tmp, OPERATION_MUL);
    tensor_vector_operator(&bn_tmp, &bn_beta, &bn_tmp, OPERATION_ADD);
    relu_operator(&bn_tmp, &relu_out);
    reduce_dim2_dim1_sum(&relu_out, &pooled_sum);
    tensor_imm_operator(&pooled_sum, &pooled_avg, 1u, wd, td, OPERATION_MUL);
    collapsed = pooled_avg;
    flatten_operator(&collapsed);
    transpose_operator(&fc_w, &fc_w_t, 0);
    gemm_operator(&collapsed, &fc_w_t_cim, &logits, &logits, 0, 0);
    tensor_vector_operator(&logits, &fc_bias, &logits_bias, OPERATION_ADD);
}
