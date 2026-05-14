#define BASE_SCRATCHPAD0    0x90000000
#define BASE_SCRATCHPAD1    0x90020000
#define BASE_SCRATCHPAD2    0x90040000
#define BASE_SCRATCHPAD3    0x90060000

#define CIMC_MODE_MEMORY    0b0000
#define CIMC_MODE_ROW4_COL1 0b0001
#define CIMC_MODE_ROW2_COL2 0b0010
#define CIMC_MODE_ROW1_COL4 0b0100

int main(void)
{
uint32_t base_addr_in       = 0xa0000000;//int8  28x28x1
uint32_t base_addr_conv1_wt = base_addr_in + 0x00005480;//int8  9x32
uint32_t base_addr_conv1_bias = base_addr_in + 0x000055a0;//int32 1x32
uint32_t base_addr_conv2_wt = base_addr_in + 0x00005620;//int8  288x64
uint32_t base_addr_conv2_bias = base_addr_in + 0x00009e20;//int32 1x64
uint32_t base_addr_conv3_wt = base_addr_in + 0x0001bf20;//int8  576x128
uint32_t base_addr_conv3_bias = base_addr_in + 0x0001c120;//int32 1x128
uint32_t base_addr_fc1_wt   = base_addr_in + 0x0001c120; //3200x256
uint32_t base_addr_fc1_bias = base_addr_in + 0x000e4120;//1x256
uint32_t base_addr_fc2_wt   = base_addr_in + 0x000e4520;//256x10
uint32_t base_addr_fc2_bias = base_addr_in + 0x000e6520;//10x1

//载入input操作int8 28x28x1
Tensor Tensor_in;
Tensor Tensor_out;

Tensor_in.TCompiler.base_addr=base_addr_in;
Tensor_in.TCompiler.dim0=9;
Tensor_in.TCompiler.dim1=676;
Tensor_in.TCompiler.dim2=1;
Tensor_in.TCompiler.if_tile = 0;
Tensor_in.TCompiler.type_data=TYPE_INT;
Tensor_in.TCompiler.wd_data=WIDTH_8;
memcpy(&Tensor_out,&Tensor_in,sizeof(Tensor));
Tensor_out.TCompiler.base_addr=BASE_SCRATCHPAD0;
load_tensor(&Tensor_in, &Tensor_out);

//切换模式
//uint32_t  BASE_CIMC_PAGE0 =(4<<17) + 0;
sw_cimc(BASE_CIMC_PAGE0, CIMC_MODE_ROW4_COL1);

//载入权重conv1_wt int8 9x32
Tensor_in.TCompiler.base_addr=base_addr_conv1_wt;
Tensor_in.TCompiler.dim0=32;
Tensor_in.TCompiler.dim1=9;
Tensor_in.TCompiler.dim2=1;
Tensor_in.TCompiler.type_data=TYPE_INT;
Tensor_in.TCompiler.wd_data=WIDTH_8;
memcpy(&Tensor_out,&Tensor_in,sizeof(Tensor));
Tensor_out.TCompiler.base_addr=BASE_CIMC_PAGE0;
load_tensor(&Tensor_in, &Tensor_out);

//载入conv1_bias至scratchpad2中 ......这里bias是INT32 1x32
Tensor_in.TCompiler.base_addr=base_addr_conv1_bias;
Tensor_in.TCompiler.dim0=32;
Tensor_in.TCompiler.dim1=1;
Tensor_in.TCompiler.dim2=1;
Tensor_in.TCompiler.type_data=TYPE_INT;
Tensor_in.TCompiler.wd_data=WIDTH_32;
memcpy(&Tensor_out,&Tensor_in,sizeof(Tensor));
Tensor_out.TCompiler.base_addr=BASE_SCRATCHPAD3;
load_tensor(&Tensor_in, &Tensor_out);

//对所有的输出,利用mov指令,将bias写入至所有的index中 int32 26x26x32
//这里由于一个scratch放不下，因此将卷积输出分成两个部分，一个放于scratchpad1，一个放于scratchpad2中
uint32_t addr_conv1_out = BASE_SCRATCHPAD1;
uint32_t addr_conv2_out = BASE_SCRATCHPAD2;
Tensor_in.TCompiler.base_addr=BASE_SCRATCHPAD3;
Tensor_in.TCompiler.dim0=32;
Tensor_in.TCompiler.dim1=1;
Tensor_in.TCompiler.dim2=1;
Tensor_in.TCompiler.type_data=TYPE_INT;
Tensor_in.TCompiler.wd_data=WIDTH_32;
memcpy(&Tensor_out,&Tensor_in,sizeof(Tensor));
mov_cfg( &Tensor_in, &Tensor_out );
for(int i=0;i<13;i++){
    for(int j=0;j<26;j++){
        TM_REG.byte_base_out = addr_conv1_out;
        mov_primitive(&TM_REG);
        addr_conv1_out += (TM_REG.cfg_size_dim0b_rem_dim0.cfg_size_dim0b<<5);
        TM_REG.byte_base_out = addr_conv2_out;
        mov_primitive(&TM_REG);
        addr_conv2_out += (TM_REG.cfg_size_dim0b_rem_dim0.cfg_size_dim0b<<5);
    }
}

//卷积第一部分
Tensor origin_tensor;
CFG_KERNEL kernel;
CONV_OPTION option;
Tensor_in.TCompiler.base_addr=BASE_SCRATCHPAD0;
Tensor_in.TCompiler.dim0=32;
Tensor_in.TCompiler.dim1=338;
Tensor_in.TCompiler.dim2=1;
Tensor_in.TCompiler.type_data=TYPE_INT;
Tensor_in.TCompiler.wd_data=WIDTH_8;
Tensor_out.TCompiler.base_addr=BASE_SCRATCHPAD1;
Tensor_out.TCompiler.dim0=32;
Tensor_out.TCompiler.dim1=338;
Tensor_out.TCompiler.dim2=1;
Tensor_out.TCompiler.type_data=TYPE_INT;
Tensor_out.TCompiler.wd_data=WIDTH_32;
origin_tensor.TCompiler.base_addr=BASE_SCRATCHPAD1;
origin_tensor.TCompiler.dim0=32;
origin_tensor.TCompiler.dim1=338;
origin_tensor.TCompiler.dim2=1;
origin_tensor.TCompiler.type_data=TYPE_INT;
origin_tensor.TCompiler.wd_data=WIDTH_32;
kernel.byte_base_wt=BASE_CIMC_PAGE0;
kernel.size_x=1;
kernel.size_y=1;
kernel.slide_x=1;
kernel.slide_y=1;
kernel.type=TYPE_INT;
kernel.wd=WIDTH_8;
option.accumulate=1;
option.activate=1;
option.dilate_x=1;
option.dilate_y=1;
option.log2trp_x=0;
option.log2trp_y=0;
option.padding_n=0;
option.padding_w=0;
option.padding_value=0;
option.shift=0;
//option.shift=0x18;
conv_operator(&Tensor_in,&Tensor_out,&origin_tensor,&kernel,&option);

Tensor Tensor_in_fp_32;
Tensor_in_fp_32.TCompiler.dim0 = 32;
Tensor_in_fp_32.TCompiler.dim1 = 26;
Tensor_in_fp_32.TCompiler.dim2 = 13;
Tensor_in_fp_32.TCompiler.if_tile = 0;
Tensor_in_fp_32.TCompiler.base_addr = BASE_SCRATCHPAD1;
Tensor_in_fp_32.TCompiler.wd_data = WIDTH_32;
Tensor_in_fp_32.TCompiler.type_data = TYPE_FP;
cast_operator(&Tensor_out,&Tensor_in_fp_32);

Tensor_in.TCompiler.base_addr=BASE_SCRATCHPAD0;
Tensor_out.TCompiler.base_addr=BASE_SCRATCHPAD2;
origin_tensor.TCompiler.base_addr=BASE_SCRATCHPAD2;
conv_operator(&Tensor_in,&Tensor_out,&origin_tensor,&kernel,&option);

//Tensor_in_fp_32.TCompiler.base_addr =  BASE_SCRATCHPAD2 + ((Tensor_in_fp_32.TChip.stride_dim2*Tensor_in_fp_32.TChip.size_dim2)<<5);
Tensor_in_fp_32.TCompiler.base_addr = BASE_SCRATCHPAD2;
cast_operator(&Tensor_out,&Tensor_in_fp_32);

vs_v_cfg(&Tensor_in_fp_32,&Tensor_in_fp_32);
for (uint32_t i = 0; i < Tensor_in_fp_32.TCompiler.dim2; i = i + 1)
    for (uint32_t j = 0; j < Tensor_in_fp_32.TCompiler.dim1; j = j + 1) {
        VP_REG.val_in2 = 0x3b7a0000;
        VP_REG.cfg_op_wd_type.cfg_op = OPERATION_MUL;
        vs_v_primitive(&VP_REG);
        VP_REG.val_in2 = 0xc3000000;
        VP_REG.cfg_op_wd_type.cfg_op = OPERATION_ADD;
        vs_v_primitive(&VP_REG);
        VP_REG.byte_base_in1 = VP_REG.byte_base_in1 + (Tensor_in_fp_32.TChip.size_dim0b<<5);
        VP_REG.byte_base_out = VP_REG.byte_base_out + (Tensor_in_fp_32.TChip.size_dim0b<<5);
    }

VP_REG.byte_base_in1 = BASE_SCRATCHPAD1;
VP_REG.byte_base_out = BASE_SCRATCHPAD1;
for (uint32_t i = 0; i < Tensor_in_fp_32.TCompiler.dim2; i = i + 1)
for (uint32_t j = 0; j < Tensor_in_fp_32.TCompiler.dim1; j = j + 1) {
        VP_REG.val_in2 = 0x3b7a0000;
        VP_REG.cfg_op_wd_type.cfg_op = OPERATION_MUL;
        vs_v_primitive(&VP_REG);
        VP_REG.val_in2 = 0xc3000000;
        VP_REG.cfg_op_wd_type.cfg_op = OPERATION_ADD;
        vs_v_primitive(&VP_REG);
        VP_REG.byte_base_in1 = VP_REG.byte_base_in1 + (Tensor_in_fp_32.TChip.size_dim0b<<5);
        VP_REG.byte_base_out = VP_REG.byte_base_in1;
    }

Tensor Tensor_out_fp_16;
memcpy(&Tensor_out_fp_16,&Tensor_in_fp_32,sizeof(Tensor));
Tensor_in_fp_32.TCompiler.base_addr = BASE_SCRATCHPAD1;
Tensor_out_fp_16.TCompiler.base_addr = BASE_SCRATCHPAD3;
Tensor_out_fp_16.TCompiler.wd_data = WIDTH_16;
cast_operator(&Tensor_in_fp_32,&Tensor_out_fp_16);

Tensor_in_fp_32.TCompiler.base_addr = BASE_SCRATCHPAD2;
Tensor_out_fp_16.TCompiler.base_addr = (Tensor_out_fp_16.TChip.stride_dim2*Tensor_out_fp_16.TChip.size_dim2)<<5;
cast_operator(&Tensor_in_fp_32,&Tensor_out_fp_16);

Tensor_in_fp_16.TCompiler.base_addr = BASE_SCRATCHPAD3;

Tensor Tensor_int_8;
Tensor_int_8.TCompiler.dim0 = 32;
Tensor_int_8.TCompiler.dim1 = 26;
Tensor_int_8.TCompiler.dim2 = 26;
Tensor_int_8.TCompiler.type_data = TYPE_INT;
Tensor_int_8.TCompiler.wd_data = WIDTH_8;
Tensor_int_8.TCompiler.if_tile = 0;
Tensor_int_8.TCompiler.base_addr = BASE_SCRATCHPAD1;
cast_operator(&Tensor_out_fp_16,&Tensor_int_8);

vs_v_cfg(&Tensor_int_8,&Tensor_int_8);
VP_REG.val_in2 = 0xffffff80;
VP_REG.cfg_op_wd_type.cfg_op = OPERATION_SUB;
for (uint32_t i = 0; i < Tensor_int_8.TCompiler.dim2; i = i + 1)
for (uint32_t j = 0; j < Tensor_int_8.TCompiler.dim1; j = j + 1) {
    vs_v_primitive(&VP_REG);
    VP_REG.byte_base_in1 = VP_REG.byte_base_in1 + (Tensor_int_8.TChip.size_dim0b<<5);
    VP_REG.byte_base_out = VP_REG.byte_base_in1;
}

//切换CIMC的寻址模式；
sw_cimc(BASE_CIMC_PAGE0, CIMC_MODE_ROW2_COL2);
//uint32_t  BASE_CIMC_PAGE1 = (4<<17) + (1<<13);

//在计算第一层卷积时，载入第二层卷积权重和bias
//载入第二层卷积权重：int8 288x64
uint32_t address_conv2_wt_gmem = base_addr_conv2_wt;
uint32_t address_conv2_wt_cim = BASE_CIMC_PAGE1 ;
Tensor_in.TCompiler.dim0=64;
Tensor_in.TCompiler.dim1=96;
Tensor_in.TCompiler.dim2=1;
Tensor_in.TCompiler.if_tile = 0;
Tensor_in.TCompiler.type_data=TYPE_INT;
Tensor_in.TCompiler.wd_data=WIDTH_8;
Tensor_out.TCompiler.base_addr=address_conv2_wt_gmem;
Tensor_out.TCompiler.dim0=64;
Tensor_out.TCompiler.dim1=96;
Tensor_out.TCompiler.dim2=1;
Tensor_out.TCompiler.if_tile = 0;
Tensor_out.TCompiler.type_data=TYPE_INT;
Tensor_out.TCompiler.wd_data=WIDTH_8;
Tensor_out.TCompiler.base_addr=address_conv2_wt_cim;
load_tensor(&Tensor_in, &Tensor_out);
for(int i=0;i<2;i++){
    DTLD_REG.byte_base_gmem += 2*3*32*32;
    DTLD_REG.byte_base_lmem  += (1<<13);
    tld_primitive(&DTLD_REG);
}

//载入第二层卷积bias  int32 1x64
Tensor_in.TCompiler.base_addr=base_addr_conv2_bias;
Tensor_in.TCompiler.dim0=64;
Tensor_in.TCompiler.dim1=1;
Tensor_in.TCompiler.dim2=1;
Tensor_in.TCompiler.type_data=TYPE_INT;
Tensor_in.TCompiler.wd_data=WIDTH_32;
memcpy(&Tensor_out,&Tensor_in,sizeof(Tensor));
Tensor_out.TCompiler.base_addr=BASE_SCRATCHPAD2+2040*32;
load_tensor(&Tensor_in, &Tensor_out);

//第二层卷积输出大小为24x24x64,bias需要分成3个部分，分别存储于scratchpad1 scratchpad2 scratchpad3中
uint32_t base_part1 = BASE_SCRATCHPAD0;
uint32_t base_part2 = BASE_SCRATCHPAD2;
uint32_t base_part3 = BASE_SCRATCHPAD3;
Tensor_in.TCompiler.base_addr=BASE_SCRATCHPAD2+2040*32;
Tensor_in.TCompiler.dim0=64;
Tensor_in.TCompiler.dim1=1;
Tensor_in.TCompiler.dim2=1;
Tensor_in.TCompiler.type_data=TYPE_INT;
Tensor_in.TCompiler.wd_data=WIDTH_32;
memcpy(&Tensor_out,&Tensor_in,sizeof(Tensor));
mov_cfg( &Tensor_in, &Tensor_out );
for(int i=0;i<8;i++){
    for(int j=0;j<24;j++){
        TM_REG.byte_base_out = base_part1;
        mov_primitive(&TM_REG);
        base_part1 += 32*8;
        TM_REG.byte_base_out = base_part2;
        mov_primitive(&TM_REG);
        base_part2 += 32*8;
        TM_REG.byte_base_out = base_part3;
        mov_primitive(&TM_REG);
        base_part3 += 32*8;
    }
}

//卷积第二部分
int if_act =0;
int if_shift =0;
//Tensor_in.TCompiler.base_addr=BASE_SCRATCHPAD0;
//Tensor_in.TCompiler.dim0=32;
//Tensor_in.TCompiler.dim1=26;
//Tensor_in.TCompiler.dim2=8;
Tensor_int_8.TCompiler.dim2 = 8;
//Tensor_in.TCompiler.type_data=TYPE_INT;
//Tensor_in.TCompiler.wd_data=WIDTH_8;
Tensor_out.TCompiler.base_addr=BASE_SCRATCHPAD0;
Tensor_out.TCompiler.dim0=64;
Tensor_out.TCompiler.dim1=24;
Tensor_out.TCompiler.dim2=8;
Tensor_out.TCompiler.type_data=TYPE_INT;
Tensor_out.TCompiler.wd_data=WIDTH_32;
origin_tensor.TCompiler.base_addr=BASE_SCRATCHPAD0;
origin_tensor.TCompiler.dim0=64;
origin_tensor.TCompiler.dim1=24;
origin_tensor.TCompiler.dim2=8;
origin_tensor.TCompiler.type_data=TYPE_INT;
origin_tensor.TCompiler.wd_data=WIDTH_32;
kernel.byte_base_wt=BASE_CIMC_PAGE1;
kernel.size_x=3;
kernel.size_y=1;
kernel.slide_x=1;
kernel.slide_y=1;
kernel.type=TYPE_INT;
kernel.wd=WIDTH_8;
option.accumulate=1;
option.activate=if_act;
option.dilate_x=1;
option.dilate_y=1;
option.log2trp_x=0;
option.log2trp_y=0;
option.padding_n=0;
option.padding_w=0;
option.padding_value=0;
option.shift=if_shift;
uint32_t address_conv2_in = BASE_SCRATCHPAD1;
for(int i=0;i<3;i++){
    // if(i==2){
    //     option.shift = 0x18;
    //     option.activate = 0;
    //     Tensor_out.TCompiler.wd_data=WIDTH_8;
    // }
    Tensor_int_8.TCompiler.base_addr = address_conv2_in;
    Tensor_out.TCompiler.base_addr=BASE_SCRATCHPAD0;
    origin_tensor.TCompiler.base_addr=BASE_SCRATCHPAD0;
    conv_operator(&Tensor_int_8,&Tensor_out,&origin_tensor,&kernel,&option);

    Tensor_int_8.TCompiler.base_addr=address_conv2_in + 32*26*8;
    Tensor_out.TCompiler.base_addr=BASE_SCRATCHPAD2;
    origin_tensor.TCompiler.base_addr=BASE_SCRATCHPAD2;
    conv_operator(&Tensor_int_8,&Tensor_out,&origin_tensor,&kernel,&option);

    Tensor_int_8.TCompiler.base_addr=address_conv2_in + 32*26*16;
    Tensor_out.TCompiler.base_addr=BASE_SCRATCHPAD3;
    origin_tensor.TCompiler.base_addr=BASE_SCRATCHPAD3;
    conv_operator(&Tensor_int_8,&Tensor_out,&origin_tensor,&kernel,&option);

    address_conv2_in = address_conv2_in + (26*32);
    kernel.byte_base_wt = kernel.byte_base_wt + (1<<13);
}

Tensor_in.TCompiler.dim0=64;
Tensor_in.TCompiler.dim1=24;
Tensor_in.TCompiler.dim2=8;
Tensor_in.TCompiler.type_data=TYPE_INT;
Tensor_in.TCompiler.wd_data=WIDTH_32;
Tensor_in.TCompiler.if_tile=0;
Tensor_in.TCompiler.base_addr=BASE_SCRATCHPAD0;

Tensor_out.TCompiler.base_addr=BASE_SCRATCHPAD1;
Tensor_out.TCompiler.dim0=64;
Tensor_out.TCompiler.dim1=24;
Tensor_out.TCompiler.dim2=8;
Tensor_out.TCompiler.type_data=TYPE_FP;
Tensor_out.TCompiler.wd_data=WIDTH_32;
Tensor_out.TCompiler.if_tile = 0;

for (uint32_t k = 0; k < 3; k = k + 1) {
    if (k == 0) Tensor_in.TCompiler.base_addr = BASE_SCRATCHPAD0;
    else if (k == 1) Tensor_in.TCompiler.base_addr = BASE_SCRATCHPAD2;
    else if (k == 2) Tensor_in.TCompiler.base_addr = BASE_SCRATCHPAD3;

    Tensor_in.TCompiler.wd_data = WIDTH_32;
    cast_operator(&Tensor_in, &Tensor_out);
    vs_v_cfg(&Tensor_out, &Tensor_out);

    for (uint32_t i = 0; i < Tensor_out.TCompiler.dim2; i = i + 1)
        for (uint32_t j = 0; j < Tensor_out.TCompiler.dim1; j = j + 1) {
            VP_REG.val_in2 = 0x3a300000;
            VP_REG.cfg_op_wd_type.cfg_op = OPERATION_MUL;
            vs_v_primitive(&VP_REG);

            VP_REG.val_in2 = 0xc3000000;
            VP_REG.cfg_op_wd_type.cfg_op = OPERATION_ADD;
            vs_v_primitive(&VP_REG);

            VP_REG.byte_base_in1 = VP_REG.byte_base_in1 + (Tensor_out.TChip.size_dim0b<<5);
            VP_REG.byte_base_out = VP_REG.byte_base_in1;
        }
}

memcpy(&Tensor_out_fp_16,&Tensor_out,sizeof(Tensor));
Tensor_in_fp_32.TCompiler.wd_data = WIDTH_16;
cast_operator(&Tensor_out,&Tensor_out_fp_16);

Tensor_in_fp_16.TCompiler.wd_data = WIDTH_8;
cast_operator(&Tensor_out_fp_16,&Tensor_in);

vs_v_cfg(&Tensor_in,&Tensor_in);
VP_REG.val_in2 = 0xffffff80;
VP_REG.cfg_op_wd_type.cfg_op = OPERATION_SUB;
for (uint32_t i = 0; i < Tensor_in.TCompiler.dim2; i = i + 1)
for (uint32_t j = 0; j < Tensor_in.TCompiler.dim1; j = j + 1) {
    vs_v_primitive(&VP_REG);
    VP_REG.byte_base_in1 = VP_REG.byte_base_in1 + (Tensor_in.TChip.size_dim0b<<5);
    VP_REG.byte_base_out = VP_REG.byte_base_in1;
}

Tensor_in.TCompiler.base_addr = BASE_SCRATCHPAD0;
mov_cfg(&Tensor_in, &Tensor_out);
TM_REG.byte_base_out = BASE_SCRATCHPAD1;
mov_primitive(&TM_REG);

// TM_REG.byte_base_in=BASE_SCRATCHPAD2;
// TM_REG.byte_base_out=BASE_SCRATCHPAD1+24*8*2*32;
// mov_primitive(&TM_REG);

// //3
// TM_REG.byte_base_in = BASE_SCRATCHPAD3;
// TM_REG.byte_base_out = BASE_SCRATCHPAD1+24*16*2*32;
// mov_primitive(&TM_REG);

//max_pooling
Tensor_in.TCompiler.base_addr=BASE_SCRATCHPAD1;
Tensor_in.TCompiler.dim0=64;
Tensor_in.TCompiler.dim1=24;
Tensor_in.TCompiler.dim2=24;
Tensor_in.TCompiler.if_tile = 0;
Tensor_in.TCompiler.type_data=TYPE_INT;
Tensor_in.TCompiler.wd_data=WIDTH_8;
Tensor_out.TCompiler.base_addr=BASE_SCRATCHPAD0;
Tensor_out.TCompiler.dim0=64;
Tensor_out.TCompiler.dim1=12;
Tensor_out.TCompiler.dim2=12;
Tensor_out.TCompiler.type_data=TYPE_INT;
Tensor_out.TCompiler.wd_data=WIDTH_8;
kernel.size_x=2;
kernel.size_y=2;
kernel.slide_x=2;
kernel.slide_y=2;
max_pooling(&Tensor_in, &kernel, &Tensor_out);

//第三层卷积分成9次计算，由于很幸运的是，在INT32的情况下，中间的
//切换CIMC的寻址模式；
sw_cimc(BASE_CIMC_PAGE0, CIMC_MODE_ROW1_COL4);
//uint32_t  BASE_CIMC_PAGE4 =(4<<17) + (4<<13);

//显然，在做第二层卷积时，我们可以同时更新第三层卷积的权重
//第三层卷积权重需要分成9次搬运int8 576x128
uint32_t base_addr_conv3_wt_gmem = base_addr_conv3_wt;
uint32_t base_addr_conv3_wt_lmem = BASE_CIMC_PAGE4;
Tensor_in.TCompiler.dim0 = 128;
Tensor_in.TCompiler.dim1=64;
Tensor_in.TCompiler.dim2=1;
Tensor_in.TCompiler.type_data=TYPE_INT;
Tensor_in.TCompiler.wd_data=WIDTH_8;
Tensor_in.TCompiler.if_tile = 0;
Tensor_in.TCompiler.base_addr=base_addr_conv3_wt_gmem;
Tensor_out.TCompiler.dim0 = 128;
Tensor_out.TCompiler.dim1=64;
Tensor_out.TCompiler.dim2=1;
Tensor_out.TCompiler.type_data=TYPE_INT;
Tensor_out.TCompiler.wd_data=WIDTH_8;
Tensor_out.TCompiler.if_tile = 0;
Tensor_out.TCompiler.base_addr=base_addr_conv3_wt_lmem;
load_tensor(&Tensor_in, &Tensor_out);
for(int i=0;i<8;i++) {
    DTLD_REG.byte_base_gmem += 4*64*32;
    DTLD_REG.byte_base_lmem  += (1<<13);
    tld_primitive(&DTLD_REG);
}

//载入第三层卷积bias  int32 1x128
Tensor_in.TCompiler.base_addr=base_addr_conv3_bias;
Tensor_in.TCompiler.dim0=128;
Tensor_in.TCompiler.dim1=1;
Tensor_in.TCompiler.dim2=1;
Tensor_in.TCompiler.type_data=TYPE_INT;
Tensor_in.TCompiler.wd_data=WIDTH_32;
Tensor_out.TCompiler.type_data=TYPE_INT;
Tensor_out.TCompiler.wd_data=WIDTH_32;
Tensor_out.TCompiler.base_addr=BASE_SCRATCHPAD2;
load_tensor(&Tensor_in, &Tensor_out);

//将载入的bias复制至conv 输出上
uint32_t addr_conv3 = BASE_SCRATCHPAD2;
Tensor_in.TCompiler.base_addr=BASE_SCRATCHPAD2;
Tensor_in.TCompiler.dim0=128;
Tensor_in.TCompiler.dim1=1;
Tensor_in.TCompiler.dim2=1;
Tensor_in.TCompiler.type_data=TYPE_INT;
Tensor_in.TCompiler.wd_data=WIDTH_32;
Tensor_out.TCompiler.base_addr=BASE_SCRATCHPAD1;
Tensor_out.TCompiler.dim0=128;
Tensor_out.TCompiler.dim1=1;
Tensor_out.TCompiler.dim2=1;
Tensor_out.TCompiler.type_data=TYPE_INT;
Tensor_out.TCompiler.wd_data=WIDTH_32;
mov_cfg( &Tensor_in, &Tensor_out );
for(int i=0;i<10;i++){
    for(int j=0;j<10;j++) {
        mov_primitive(&TM_REG);
        TM_REG.byte_base_out += 32*16;
    }
}

//卷积第三部分
//开始进行卷积操作
if_act =0;
if_shift =0;
uint32_t addr_conv3_wt = BASE_CIMC_PAGE4;
Tensor_in.TCompiler.dim0=64;
Tensor_in.TCompiler.dim1=10;
Tensor_in.TCompiler.dim2=10;
Tensor_in.TCompiler.type_data=TYPE_INT;
Tensor_in.TCompiler.wd_data=WIDTH_8;
Tensor_in.TCompiler.if_tile = 1;
Tensor_in.TCompiler.dim0_ori = 64;
Tensor_in.TCompiler.dim1_ori = 12;
Tensor_in.TCompiler.dim2_ori = 12;
Tensor_out.TCompiler.base_addr=BASE_SCRATCHPAD1;
Tensor_out.TCompiler.dim0=128;
Tensor_out.TCompiler.dim1=10;
Tensor_out.TCompiler.dim2=10;
Tensor_out.TCompiler.type_data=TYPE_INT;
Tensor_out.TCompiler.wd_data=WIDTH_32;
origin_tensor.TCompiler.base_addr=BASE_SCRATCHPAD1;
origin_tensor.TCompiler.dim0=128;
origin_tensor.TCompiler.dim1=10;
origin_tensor.TCompiler.dim2=10;
origin_tensor.TCompiler.type_data=TYPE_INT;
origin_tensor.TCompiler.wd_data=WIDTH_32;
kernel.byte_base_wt=addr_conv3_wt;
kernel.size_x=1;
kernel.size_y=1;
kernel.slide_x=1;
kernel.slide_y=1;
kernel.type=TYPE_INT;
kernel.wd=WIDTH_8;
option.accumulate=1;
option.activate=0;
option.shift = 0;
option.dilate_x=1;
option.dilate_y=1;
option.log2trp_x=0;
option.log2trp_y=0;
option.padding_n=0;
option.padding_w=0;
option.padding_value=0;
uint32_t addr_conv3_in;

for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
    // if(i==2 && j==2){
    //     option.activate = 1;
    //     option.shift = 0x18;
    //     Tensor_out.TCompiler.wd_data=WIDTH_8;
    // }
        Tensor_in.TCompiler.base_addr = BASE_SCRATCHPAD0 + (((i*12+j)*2)<<5);
        conv_operator(&Tensor_in,&Tensor_out,&origin_tensor,&kernel,&option);
        kernel.byte_base_wt += (1<<13);
    }
}

//输出卷积换个位置
Tensor_in.TCompiler.base_addr=BASE_SCRATCHPAD1;
Tensor_in.TCompiler.dim0=128;
Tensor_in.TCompiler.dim1=10;
Tensor_in.TCompiler.dim2=10;
Tensor_in.TCompiler.type_data=TYPE_INT;
Tensor_in.TCompiler.wd_data=WIDTH_32;
Tensor_in.TCompiler.if_tile=0;
Tensor_out.TCompiler.base_addr=BASE_SCRATCHPAD0;
Tensor_out.TCompiler.dim0=128;
Tensor_out.TCompiler.dim1=10;
Tensor_out.TCompiler.dim2=10;
Tensor_out.TCompiler.type_data=TYPE_FP;
Tensor_out.TCompiler.wd_data=WIDTH_32;
Tensor_out.TCompiler.if_tile = 0;
cast_operator(&Tensor_in,&Tensor_out);
vs_v_cfg(&Tensor_out,&Tensor_out);

for (uint32_t i = 0; i < Tensor_out.TCompiler.dim2; i = i + 1)
for (uint32_t j = 0; j < Tensor_out.TCompiler.dim1; j = j + 1) {
    VP_REG.val_in2 = 0x3a300000;
    VP_REG.cfg_op_wd_type.cfg_op = OPERATION_MUL;
    vs_v_primitive(&VP_REG);
    VP_REG.val_in2 = 0xc3000000;
    VP_REG.cfg_op_wd_type.cfg_op = OPERATION_ADD;
    vs_v_primitive(&VP_REG);
    VP_REG.byte_base_in1 = VP_REG.byte_base_in1 + (Tensor_out.TChip.size_dim0b<<5);
    VP_REG.byte_base_out = VP_REG.byte_base_in1;
}

memcpy(&Tensor_out_fp_16,&Tensor_out,sizeof(Tensor));
Tensor_in_fp_32.TCompiler.wd_data = WIDTH_16;
cast_operator(&Tensor_out,&Tensor_out_fp_16);

Tensor_in_fp_16.TCompiler.wd_data = WIDTH_8;
cast_operator(&Tensor_out_fp_16,&Tensor_in);

vs_v_cfg(&Tensor_in,&Tensor_in);
VP_REG.val_in2 = 0xffffff80;
VP_REG.cfg_op_wd_type.cfg_op = OPERATION_SUB;
for (uint32_t i = 0; i < Tensor_in.TCompiler.dim2; i = i + 1)
for (uint32_t j = 0; j < Tensor_in.TCompiler.dim1; j = j + 1) {
    vs_v_primitive(&VP_REG);
    VP_REG.byte_base_in1 = VP_REG.byte_base_in1 + (Tensor_in.TChip.size_dim0b<<5);
    VP_REG.byte_base_out = VP_REG.byte_base_in1;
}
}
