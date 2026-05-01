//! Cross-architecture parity test for butteraugli SIMD implementations.
//!
//! Verifies that butteraugli produces identical scores on all architectures by
//! comparing against reference scores captured on x86_64 (AVX2). Each score is
//! stored as the raw u64 bit pattern of an f64, enabling bit-exact verification.
//!
//! Run with: `cargo test --test cross_arch_parity`
//! Cross-test: `cross test -p butteraugli --test cross_arch_parity --target aarch64-unknown-linux-gnu`

mod common;

use butteraugli::{ButteraugliParams, Img, butteraugli};
use common::generators::{generate_image_pair, parse_dimensions, rgb_bytes_to_pixels};

/// Minimum dimension requirement for butteraugli
const MIN_DIMENSION: usize = 8;

/// Maximum allowed relative difference between architectures.
///
/// FMA rounding differences between x86 AVX2 and ARM NEON can produce
/// tiny variations (~7e-6 in f32 intermediates). After accumulation into
/// the final f64 score, we allow up to 0.01% (1e-4) relative difference.
/// Low-scoring images (quantize distortions, ~0.4 score) amplify tiny
/// absolute differences into ~5e-5 relative, so 1e-4 provides headroom.
/// Real algorithm bugs would show >1% difference.
const MAX_RELATIVE_DIFF: f64 = 1e-4;

/// Reference scores captured on x86_64 with AVX2, stored as f64 bit patterns.
/// Each entry is (test_case_name, f64::to_bits() value).
const X86_REFERENCE_SCORES: &[(&str, u64)] = &[
    ("uniform_gray_128_shift_10_8x8", 0x402F88E5E0000000),
    ("uniform_gray_128_shift_50_8x8", 0x4052A74100000000),
    ("uniform_gray_128_shift_10_9x9", 0x402F88E1C0000000),
    ("uniform_gray_128_shift_50_9x9", 0x4052A742E0000000),
    ("uniform_gray_128_shift_10_15x15", 0x4035492E80000000),
    ("uniform_gray_128_shift_50_15x15", 0x40592E95C0000000),
    ("uniform_gray_128_shift_10_16x16", 0x4035493920000000),
    ("uniform_gray_128_shift_50_16x16", 0x40592E9F80000000),
    ("uniform_gray_128_shift_10_17x17", 0x40354928C0000000),
    ("uniform_gray_128_shift_50_17x17", 0x40592E9400000000),
    ("uniform_gray_128_shift_10_23x23", 0x4035492780000000),
    ("uniform_gray_128_shift_50_23x23", 0x40592E8EC0000000),
    ("uniform_gray_128_shift_10_24x24", 0x4035492340000000),
    ("uniform_gray_128_shift_50_24x24", 0x40592E88A0000000),
    ("uniform_gray_128_shift_10_31x31", 0x4035492E80000000),
    ("uniform_gray_128_shift_50_31x31", 0x40592E9340000000),
    ("uniform_gray_128_shift_10_32x32", 0x4035492E40000000),
    ("uniform_gray_128_shift_50_32x32", 0x40592E9500000000),
    ("uniform_red_shift_20_16x16", 0x403F24C500000000),
    ("uniform_green_shift_20_16x16", 0x4044F409E0000000),
    ("uniform_blue_shift_20_16x16", 0x4028B48A40000000),
    ("uniform_red_shift_20_23x23", 0x403F24C060000000),
    ("uniform_green_shift_20_23x23", 0x4044F3F5A0000000),
    ("uniform_blue_shift_20_23x23", 0x4028B488C0000000),
    ("uniform_red_shift_20_32x32", 0x403F24BD80000000),
    ("uniform_green_shift_20_32x32", 0x4044F3F700000000),
    ("uniform_blue_shift_20_32x32", 0x4028B48640000000),
    ("gradient_h_shift_15_8x8", 0x4016ADA460000000),
    ("gradient_v_shift_15_8x8", 0x4016ADA360000000),
    ("gradient_h_shift_15_9x9", 0x401746AAE0000000),
    ("gradient_v_shift_15_9x9", 0x401746AC60000000),
    ("gradient_h_shift_15_15x15", 0x4021790A00000000),
    ("gradient_v_shift_15_15x15", 0x4021790BC0000000),
    ("gradient_h_shift_15_16x16", 0x40223EBDA0000000),
    ("gradient_v_shift_15_16x16", 0x40223EBB40000000),
    ("gradient_h_shift_15_17x17", 0x4023023D00000000),
    ("gradient_v_shift_15_17x17", 0x4023023C40000000),
    ("gradient_h_shift_15_23x23", 0x40270DA020000000),
    ("gradient_v_shift_15_23x23", 0x40270DA000000000),
    ("gradient_h_shift_15_24x24", 0x402874DA40000000),
    ("gradient_v_shift_15_24x24", 0x402874DC20000000),
    ("gradient_h_shift_15_31x31", 0x402981BD40000000),
    ("gradient_v_shift_15_31x31", 0x402981BB00000000),
    ("gradient_h_shift_15_32x32", 0x402AC30740000000),
    ("gradient_v_shift_15_32x32", 0x402AC30060000000),
    ("gradient_h_shift_15_33x33", 0x402EC5B8A0000000),
    ("gradient_v_shift_15_33x33", 0x402EC5BA40000000),
    ("gradient_h_shift_15_47x47", 0x402F1365C0000000),
    ("gradient_v_shift_15_47x47", 0x402F136D00000000),
    ("gradient_diag_shift_20_16x16", 0x402BCB33E0000000),
    ("gradient_diag_shift_20_23x31", 0x4033D01E80000000),
    ("gradient_diag_shift_20_32x32", 0x4033B1FCE0000000),
    ("gradient_diag_shift_20_47x33", 0x4034C29EA0000000),
    ("color_gradient_shift_10_16x16", 0x40176B8F40000000),
    ("color_gradient_shift_10_23x23", 0x401E011100000000),
    ("color_gradient_shift_10_32x32", 0x4021FA2000000000),
    ("color_gradient_shift_10_33x47", 0x402428E2C0000000),
    ("checkerboard_vs_inverse_1px_8x8", 0x40197F9420000000),
    ("checkerboard_vs_inverse_2px_8x8", 0x402438AA40000000),
    ("checkerboard_shift_10_8x8", 0x400E98AFC0000000),
    ("checkerboard_vs_inverse_1px_15x15", 0x402142B400000000),
    ("checkerboard_vs_inverse_2px_15x15", 0x4023D0C240000000),
    ("checkerboard_shift_10_15x15", 0x401433EB00000000),
    ("checkerboard_vs_inverse_1px_16x16", 0x4018BE5400000000),
    ("checkerboard_vs_inverse_2px_16x16", 0x4023CC7D60000000),
    ("checkerboard_shift_10_16x16", 0x40144558C0000000),
    ("checkerboard_vs_inverse_1px_23x23", 0x402120B960000000),
    ("checkerboard_vs_inverse_2px_23x23", 0x4023C4C7C0000000),
    ("checkerboard_shift_10_23x23", 0x4014332600000000),
    ("checkerboard_vs_inverse_1px_32x32", 0x4018BEC780000000),
    ("checkerboard_vs_inverse_2px_32x32", 0x4023C386E0000000),
    ("checkerboard_shift_10_32x32", 0x4014427720000000),
    ("checkerboard_vs_inverse_1px_33x33", 0x4021221AE0000000),
    ("checkerboard_vs_inverse_2px_33x33", 0x4023C3B2C0000000),
    ("checkerboard_shift_10_33x33", 0x401432DA00000000),
    ("checkerboard_vs_inverse_4px_32x32", 0x4043E1A740000000),
    ("checkerboard_vs_inverse_8px_32x32", 0x4052B31540000000),
    ("checkerboard_vs_inverse_4px_47x47", 0x4043B78780000000),
    ("checkerboard_vs_inverse_8px_47x47", 0x40523FD540000000),
    ("checkerboard_vs_inverse_4px_64x64", 0x4043E48120000000),
    ("checkerboard_vs_inverse_8px_64x64", 0x4052B45B00000000),
    ("stripes_h_2px_shift_15_16x16", 0x401FC5A9A0000000),
    ("stripes_v_2px_shift_15_16x16", 0x401FC5AB40000000),
    ("stripes_h_2px_shift_15_23x23", 0x401EE14C80000000),
    ("stripes_v_2px_shift_15_23x23", 0x401EE14D00000000),
    ("stripes_h_2px_shift_15_32x32", 0x401FCA0180000000),
    ("stripes_v_2px_shift_15_32x32", 0x401FC9FFE0000000),
    ("stripes_h_2px_shift_15_33x47", 0x401EE4AB80000000),
    ("stripes_v_2px_shift_15_33x47", 0x401EC60940000000),
    ("sine_1x1_shift_10_32x32", 0x401B287360000000),
    ("sine_2x2_shift_10_32x32", 0x40183D6AE0000000),
    ("sine_4x4_shift_10_32x32", 0x40173B9CA0000000),
    ("sine_1x1_shift_10_33x33", 0x401BF8FD40000000),
    ("sine_2x2_shift_10_33x33", 0x40185F3680000000),
    ("sine_4x4_shift_10_33x33", 0x40173C6580000000),
    ("sine_1x1_shift_10_47x47", 0x4021DEAE40000000),
    ("sine_2x2_shift_10_47x47", 0x401A3BAF60000000),
    ("sine_4x4_shift_10_47x47", 0x40178B2C40000000),
    ("sine_1x1_shift_10_64x64", 0x40232A3C20000000),
    ("sine_2x2_shift_10_64x64", 0x401D126DC0000000),
    ("sine_4x4_shift_10_64x64", 0x401852FA80000000),
    ("radial_shift_15_16x16", 0x4022C8E480000000),
    ("radial_shift_15_23x23", 0x402312D340000000),
    ("radial_shift_15_32x32", 0x40251430E0000000),
    ("radial_shift_15_47x47", 0x4028DD33E0000000),
    ("edge_v_shift_10_16x16", 0x4015CF3AA0000000),
    ("edge_h_shift_10_16x16", 0x4015CF3B80000000),
    ("edge_v_vs_blur_16x16", 0x40111B9E40000000),
    ("edge_v_shift_10_23x31", 0x40169B08E0000000),
    ("edge_h_shift_10_23x31", 0x40184584C0000000),
    ("edge_v_vs_blur_23x31", 0x4016545060000000),
    ("edge_v_shift_10_32x32", 0x40187B0EE0000000),
    ("edge_h_shift_10_32x32", 0x40187B1340000000),
    ("edge_v_vs_blur_32x32", 0x401217A620000000),
    ("edge_v_shift_10_47x33", 0x4026752340000000),
    ("edge_h_shift_10_47x33", 0x4018BBC100000000),
    ("edge_v_vs_blur_47x33", 0x4016C28D00000000),
    ("random_seed0_shift_10_16x16", 0x40146B5C80000000),
    ("random_seed0_noise_20_16x16", 0x3FFDD9C4C0000000),
    ("random_seed0_shift_10_23x23", 0x4013E52A20000000),
    ("random_seed0_noise_20_23x23", 0x3FFADC2900000000),
    ("random_seed0_shift_10_32x32", 0x4013F50400000000),
    ("random_seed0_noise_20_32x32", 0x3FFC725880000000),
    ("random_seed0_shift_10_33x47", 0x40144AAC00000000),
    ("random_seed0_noise_20_33x47", 0x3FF9F065E0000000),
    ("random_seed0_shift_10_47x33", 0x40145BBE00000000),
    ("random_seed0_noise_20_47x33", 0x40005FB540000000),
    ("random_seed1_shift_10_16x16", 0x401427AAC0000000),
    ("random_seed1_noise_20_16x16", 0x3FFD02EA80000000),
    ("random_seed1_shift_10_23x23", 0x40141F6580000000),
    ("random_seed1_noise_20_23x23", 0x3FF9AD7960000000),
    ("random_seed1_shift_10_32x32", 0x401405A000000000),
    ("random_seed1_noise_20_32x32", 0x3FFD1A7440000000),
    ("random_seed1_shift_10_33x47", 0x40144515C0000000),
    ("random_seed1_noise_20_33x47", 0x3FFC265280000000),
    ("random_seed1_shift_10_47x33", 0x4014197F40000000),
    ("random_seed1_noise_20_47x33", 0x400142CC20000000),
    ("random_seed2_shift_10_16x16", 0x4013E89E80000000),
    ("random_seed2_noise_20_16x16", 0x3FF5ECAA80000000),
    ("random_seed2_shift_10_23x23", 0x4013D73BA0000000),
    ("random_seed2_noise_20_23x23", 0x3FF913D8C0000000),
    ("random_seed2_shift_10_32x32", 0x4014578100000000),
    ("random_seed2_noise_20_32x32", 0x3FFBB54440000000),
    ("random_seed2_shift_10_33x47", 0x40141684C0000000),
    ("random_seed2_noise_20_33x47", 0x40006755C0000000),
    ("random_seed2_shift_10_47x33", 0x4014348F00000000),
    ("random_seed2_noise_20_47x33", 0x3FFEE4A940000000),
    ("random_seed3_shift_10_16x16", 0x401430C460000000),
    ("random_seed3_noise_20_16x16", 0x3FF5BBAB40000000),
    ("random_seed3_shift_10_23x23", 0x4014211380000000),
    ("random_seed3_noise_20_23x23", 0x3FFC40F700000000),
    ("random_seed3_shift_10_32x32", 0x40147E60A0000000),
    ("random_seed3_noise_20_32x32", 0x3FFEDE96A0000000),
    ("random_seed3_shift_10_33x47", 0x4014835580000000),
    ("random_seed3_noise_20_33x47", 0x3FFDB69B00000000),
    ("random_seed3_shift_10_47x33", 0x40144275E0000000),
    ("random_seed3_noise_20_47x33", 0x3FFA6CF020000000),
    ("random_seed4_shift_10_16x16", 0x4014A94480000000),
    ("random_seed4_noise_20_16x16", 0x3FF6F3BC00000000),
    ("random_seed4_shift_10_23x23", 0x40147856E0000000),
    ("random_seed4_noise_20_23x23", 0x3FF8F1F5E0000000),
    ("random_seed4_shift_10_32x32", 0x40141336C0000000),
    ("random_seed4_noise_20_32x32", 0x3FFB77C860000000),
    ("random_seed4_shift_10_33x47", 0x4013F41100000000),
    ("random_seed4_noise_20_33x47", 0x3FFDBB7340000000),
    ("random_seed4_shift_10_47x33", 0x40145CC380000000),
    ("random_seed4_noise_20_47x33", 0x3FFB454720000000),
    ("random_mid_contrast_1.2_32x32", 0x400A0D1340000000),
    ("random_mid_gamma_0.9_32x32", 0x400E6F90A0000000),
    ("random_mid_blur_32x32", 0x4022CD2500000000),
    ("random_mid_quantize_32_32x32", 0x3FDB8C61A0000000),
    ("random_mid_contrast_1.2_47x47", 0x400C7D1980000000),
    ("random_mid_gamma_0.9_47x47", 0x400E96EB00000000),
    ("random_mid_blur_47x47", 0x4022F254E0000000),
    ("random_mid_quantize_32_47x47", 0x3FDC3C0BC0000000),
    ("random_mid_contrast_1.2_64x64", 0x400A6A5C80000000),
    ("random_mid_gamma_0.9_64x64", 0x400ED05560000000),
    ("random_mid_blur_64x64", 0x40233D3F40000000),
    ("random_mid_quantize_32_64x64", 0x3FDF8E0D80000000),
    ("color_grad_channel_swap_16x16", 0x4041EB8C40000000),
    ("color_grad_hue_shift_16x16", 0x4048263F00000000),
    ("color_grad_channel_swap_23x23", 0x4049D56B00000000),
    ("color_grad_hue_shift_23x23", 0x4050B28F00000000),
    ("color_grad_channel_swap_32x32", 0x4050A838A0000000),
    ("color_grad_hue_shift_32x32", 0x4056015C80000000),
    ("color_grad_channel_swap_47x33", 0x4055FC0240000000),
    ("color_grad_hue_shift_47x33", 0x405B88E600000000),
    ("random_color_channel_swap_32x32", 0x401948CE40000000),
    ("random_color_hue_shift_32x32", 0x402436CCC0000000),
    ("random_color_channel_swap_47x47", 0x4017925880000000),
    ("random_color_hue_shift_47x47", 0x4024FA6DE0000000),
];

/// Reference pnorm_3 (libjxl 3-norm) values, x86_64 AVX2 baseline,
/// stored as f64 bit patterns. Indexed by name; lookup is linear.
///
/// Captured by running `cargo test --test cross_arch_parity capture_pnorm_3 -- --ignored --nocapture`
/// and pasting the output here. Re-capture only when the algorithm changes
/// intentionally (and call it out in the commit).
const X86_REFERENCE_PNORM_3: &[(&str, u64)] = &[
    ("uniform_gray_128_shift_10_8x8", 0x402F88D89E041793),
    ("uniform_gray_128_shift_50_8x8", 0x4052A745600034AB),
    ("uniform_gray_128_shift_10_9x9", 0x402F88CD0461B581),
    ("uniform_gray_128_shift_50_9x9", 0x4052A73E148C25BB),
    ("uniform_gray_128_shift_10_15x15", 0x4035492AEAF55CF5),
    ("uniform_gray_128_shift_50_15x15", 0x40592E9AF15B31F1),
    ("uniform_gray_128_shift_10_16x16", 0x4035492C7A01ECB5),
    ("uniform_gray_128_shift_50_16x16", 0x40592E9B93C120F9),
    ("uniform_gray_128_shift_10_17x17", 0x4035492083C9E7F6),
    ("uniform_gray_128_shift_50_17x17", 0x40592E8F5C96B494),
    ("uniform_gray_128_shift_10_23x23", 0x403549155D5ACB71),
    ("uniform_gray_128_shift_50_23x23", 0x40592E819FA5FA05),
    ("uniform_gray_128_shift_10_24x24", 0x4035490FE6A0C779),
    ("uniform_gray_128_shift_50_24x24", 0x40592E7C01498028),
    ("uniform_gray_128_shift_10_31x31", 0x4035491D8DBD6FF3),
    ("uniform_gray_128_shift_50_31x31", 0x40592E8C706EB2F5),
    ("uniform_gray_128_shift_10_32x32", 0x4035491E236E4424),
    ("uniform_gray_128_shift_50_32x32", 0x40592E8CD6D561CD),
    ("uniform_red_shift_20_16x16", 0x403F24B5B526B813),
    ("uniform_green_shift_20_16x16", 0x4044F3F9B30199EB),
    ("uniform_blue_shift_20_16x16", 0x4028B47830A304A9),
    ("uniform_red_shift_20_23x23", 0x403F24B2B7B95DC3),
    ("uniform_green_shift_20_23x23", 0x4044F3EBA3557598),
    ("uniform_blue_shift_20_23x23", 0x4028B46D82DCCE15),
    ("uniform_red_shift_20_32x32", 0x403F24AC8E5730F9),
    ("uniform_green_shift_20_32x32", 0x4044F3F13AA70326),
    ("uniform_blue_shift_20_32x32", 0x4028B475EF7CD32B),
    ("gradient_h_shift_15_8x8", 0x4016433792236B0D),
    ("gradient_v_shift_15_8x8", 0x40164339A0AF642B),
    ("gradient_h_shift_15_9x9", 0x4016791566C76510),
    ("gradient_v_shift_15_9x9", 0x401679162F0AB13F),
    ("gradient_h_shift_15_15x15", 0x4020B61832D249A4),
    ("gradient_v_shift_15_15x15", 0x4020B61804FCF0F0),
    ("gradient_h_shift_15_16x16", 0x4021470A02B7A761),
    ("gradient_v_shift_15_16x16", 0x4021470993E23BF1),
    ("gradient_h_shift_15_17x17", 0x4021AF307A0C6EDD),
    ("gradient_v_shift_15_17x17", 0x4021AF2EE941883F),
    ("gradient_h_shift_15_23x23", 0x4024EF7DB5352FA0),
    ("gradient_v_shift_15_23x23", 0x4024EF7F6F04E13A),
    ("gradient_h_shift_15_24x24", 0x4025C5411EAD4287),
    ("gradient_v_shift_15_24x24", 0x4025C54128611269),
    ("gradient_h_shift_15_31x31", 0x40273D02750016CC),
    ("gradient_v_shift_15_31x31", 0x40273D024D7A06C5),
    ("gradient_h_shift_15_32x32", 0x4027FDD67D3E0601),
    ("gradient_v_shift_15_32x32", 0x4027FDD5E1857488),
    ("gradient_h_shift_15_33x33", 0x402AE6F18BDBAEC8),
    ("gradient_v_shift_15_33x33", 0x402AE6F300CAAA31),
    ("gradient_h_shift_15_47x47", 0x402B39EA6F7AB6FB),
    ("gradient_v_shift_15_47x47", 0x402B39F01197FC5F),
    ("gradient_diag_shift_20_16x16", 0x402A3CC264BA0737),
    ("gradient_diag_shift_20_23x31", 0x40313689D73CA0DC),
    ("gradient_diag_shift_20_32x32", 0x4031C95EB90BCFA9),
    ("gradient_diag_shift_20_47x33", 0x403242EBD345A280),
    ("color_gradient_shift_10_16x16", 0x4016772F158CAC57),
    ("color_gradient_shift_10_23x23", 0x401AEE3574A604C0),
    ("color_gradient_shift_10_32x32", 0x401FEF2F2E8B79A1),
    ("color_gradient_shift_10_33x47", 0x402197EBE77C16CF),
    ("checkerboard_vs_inverse_1px_8x8", 0x40154C41D2C0DA39),
    ("checkerboard_vs_inverse_2px_8x8", 0x401B32FB88CE5468),
    ("checkerboard_shift_10_8x8", 0x400E92B45D328893),
    ("checkerboard_vs_inverse_1px_15x15", 0x401BEC9E83ACE89B),
    ("checkerboard_vs_inverse_2px_15x15", 0x401D10BEC08861FD),
    ("checkerboard_shift_10_15x15", 0x401429AA5D08937E),
    ("checkerboard_vs_inverse_1px_16x16", 0x4015DF208C379BE7),
    ("checkerboard_vs_inverse_2px_16x16", 0x401DABC0284B02D1),
    ("checkerboard_shift_10_16x16", 0x401438E893D58455),
    ("checkerboard_vs_inverse_1px_23x23", 0x401A883C8059DE04),
    ("checkerboard_vs_inverse_2px_23x23", 0x401EE5928D0C6E6D),
    ("checkerboard_shift_10_23x23", 0x4014250092812C30),
    ("checkerboard_vs_inverse_1px_32x32", 0x40175CB8265D2315),
    ("checkerboard_vs_inverse_2px_32x32", 0x401FD641BFCD8C09),
    ("checkerboard_shift_10_32x32", 0x401428F17D5EDCB0),
    ("checkerboard_vs_inverse_1px_33x33", 0x401964819FBEC9C8),
    ("checkerboard_vs_inverse_2px_33x33", 0x401FEDA53370CCDC),
    ("checkerboard_shift_10_33x33", 0x4014229B259E652D),
    ("checkerboard_vs_inverse_4px_32x32", 0x40392D9EB61F0944),
    ("checkerboard_vs_inverse_8px_32x32", 0x404A8DBB30029F5B),
    ("checkerboard_vs_inverse_4px_47x47", 0x40376DD7F20095F7),
    ("checkerboard_vs_inverse_8px_47x47", 0x4049576EC61E0B85),
    ("checkerboard_vs_inverse_4px_64x64", 0x4037E4DCD9BD0B13),
    ("checkerboard_vs_inverse_8px_64x64", 0x404973920D9BAB13),
    ("stripes_h_2px_shift_15_16x16", 0x401F369FFABC80E0),
    ("stripes_v_2px_shift_15_16x16", 0x401F369EFC5A59A0),
    ("stripes_h_2px_shift_15_23x23", 0x401EC9E266BFA60B),
    ("stripes_v_2px_shift_15_23x23", 0x401EC9E15F5D9F29),
    ("stripes_h_2px_shift_15_32x32", 0x401EFF12C6D9108D),
    ("stripes_v_2px_shift_15_32x32", 0x401EFF13346AC8A3),
    ("stripes_h_2px_shift_15_33x47", 0x401ECBEE04BD1DF4),
    ("stripes_v_2px_shift_15_33x47", 0x401EBE138B741A38),
    ("sine_1x1_shift_10_32x32", 0x4018309B0FF6E741),
    ("sine_2x2_shift_10_32x32", 0x4016F72983F6BEF4),
    ("sine_4x4_shift_10_32x32", 0x4017200C096791BD),
    ("sine_1x1_shift_10_33x33", 0x40188A1133D55C2D),
    ("sine_2x2_shift_10_33x33", 0x40170418BF980539),
    ("sine_4x4_shift_10_33x33", 0x4017242C8907424B),
    ("sine_1x1_shift_10_47x47", 0x401CA6A349BE70F3),
    ("sine_2x2_shift_10_47x47", 0x4017583CCCD9C2EB),
    ("sine_4x4_shift_10_47x47", 0x40173A7A9E9591F8),
    ("sine_1x1_shift_10_64x64", 0x401FDD8BD73BFF80),
    ("sine_2x2_shift_10_64x64", 0x4018D985CE8A29E5),
    ("sine_4x4_shift_10_64x64", 0x401724445B177033),
    ("radial_shift_15_16x16", 0x4022853A24D22388),
    ("radial_shift_15_23x23", 0x4022CC43E76F30B9),
    ("radial_shift_15_32x32", 0x40244FB1B832CCE8),
    ("radial_shift_15_47x47", 0x40277A27FCC59B67),
    ("edge_v_shift_10_16x16", 0x40153DF08B3446B0),
    ("edge_h_shift_10_16x16", 0x40153DF424F11E80),
    ("edge_v_vs_blur_16x16", 0x4005C426E2EC7BA8),
    ("edge_v_shift_10_23x31", 0x4015E9B5E43B83E8),
    ("edge_h_shift_10_23x31", 0x401732105555FBC9),
    ("edge_v_vs_blur_23x31", 0x400B9B67B7AC9148),
    ("edge_v_shift_10_32x32", 0x40175A8034425378),
    ("edge_h_shift_10_32x32", 0x40175A7F676340A7),
    ("edge_v_vs_blur_32x32", 0x40053CBE71BCF6E6),
    ("edge_v_shift_10_47x33", 0x40210D6594CFE2FE),
    ("edge_h_shift_10_47x33", 0x40179FD6D8D6C541),
    ("edge_v_vs_blur_47x33", 0x400900898ACE0A83),
    ("random_seed0_shift_10_16x16", 0x401405CAB50A04BC),
    ("random_seed0_noise_20_16x16", 0x3FF68C9AB07E5818),
    ("random_seed0_shift_10_23x23", 0x4013B4AD60CF7EA1),
    ("random_seed0_noise_20_23x23", 0x3FF3773D48986903),
    ("random_seed0_shift_10_32x32", 0x4013818EF6E20F5B),
    ("random_seed0_noise_20_32x32", 0x3FF38261BCF42A33),
    ("random_seed0_shift_10_33x47", 0x4013A643388E023E),
    ("random_seed0_noise_20_33x47", 0x3FF22F0692AA4763),
    ("random_seed0_shift_10_47x33", 0x4013B6960ADF4749),
    ("random_seed0_noise_20_47x33", 0x3FF3244819893041),
    ("random_seed1_shift_10_16x16", 0x4013B79230DAA037),
    ("random_seed1_noise_20_16x16", 0x3FF2B1907F17F6D5),
    ("random_seed1_shift_10_23x23", 0x4013A057FA09429F),
    ("random_seed1_noise_20_23x23", 0x3FF1EB675E054144),
    ("random_seed1_shift_10_32x32", 0x4013AFF71123DC69),
    ("random_seed1_noise_20_32x32", 0x3FF28A78E78002C6),
    ("random_seed1_shift_10_33x47", 0x4013BBBE31A7DC43),
    ("random_seed1_noise_20_33x47", 0x3FF28ABAAD2901B1),
    ("random_seed1_shift_10_47x33", 0x4013C6B58D66EB50),
    ("random_seed1_noise_20_47x33", 0x3FF3F6A13FE0F1E0),
    ("random_seed2_shift_10_16x16", 0x4013AFA658706319),
    ("random_seed2_noise_20_16x16", 0x3FEFFB08B5C4BD99),
    ("random_seed2_shift_10_23x23", 0x4013A175AA5B9DDD),
    ("random_seed2_noise_20_23x23", 0x3FF208DBE4E28939),
    ("random_seed2_shift_10_32x32", 0x4013E0FF97294BA0),
    ("random_seed2_noise_20_32x32", 0x3FF20D41C72F2653),
    ("random_seed2_shift_10_33x47", 0x4013D28C6DB654A0),
    ("random_seed2_noise_20_33x47", 0x3FF2DBC6F2476D15),
    ("random_seed2_shift_10_47x33", 0x4013DFC3FE2C4535),
    ("random_seed2_noise_20_47x33", 0x3FF24BB84D20C934),
    ("random_seed3_shift_10_16x16", 0x4013DD06A386524E),
    ("random_seed3_noise_20_16x16", 0x3FEEDA465F174060),
    ("random_seed3_shift_10_23x23", 0x4013F356B3B867ED),
    ("random_seed3_noise_20_23x23", 0x3FF28704A85B45BA),
    ("random_seed3_shift_10_32x32", 0x4013F66E4D1541C3),
    ("random_seed3_noise_20_32x32", 0x3FF22B39CAF68A34),
    ("random_seed3_shift_10_33x47", 0x4013FE6FDAFA340C),
    ("random_seed3_noise_20_33x47", 0x3FF2379EBFCF2F35),
    ("random_seed3_shift_10_47x33", 0x4013F904056AC36B),
    ("random_seed3_noise_20_47x33", 0x3FF29C1416818BD9),
    ("random_seed4_shift_10_16x16", 0x40144DAC68506545),
    ("random_seed4_noise_20_16x16", 0x3FF1C64FDA60C88B),
    ("random_seed4_shift_10_23x23", 0x4013BC3E50E76E0C),
    ("random_seed4_noise_20_23x23", 0x3FF0D5830ADC56B7),
    ("random_seed4_shift_10_32x32", 0x40138274ABAA443D),
    ("random_seed4_noise_20_32x32", 0x3FF274797597FE17),
    ("random_seed4_shift_10_33x47", 0x40139B6F3F4AA52D),
    ("random_seed4_noise_20_33x47", 0x3FF2CEB400DA48AC),
    ("random_seed4_shift_10_47x33", 0x401389D0F389A8ED),
    ("random_seed4_noise_20_47x33", 0x3FF24DE4C5B12820),
    ("random_mid_contrast_1.2_32x32", 0x4007FD6B4AD7BB51),
    ("random_mid_gamma_0.9_32x32", 0x400DD9744FC8D6DD),
    ("random_mid_blur_32x32", 0x4020903AE204AC70),
    ("random_mid_quantize_32_32x32", 0x3FD49EBD15A96A9F),
    ("random_mid_contrast_1.2_47x47", 0x4007ACDCC21152A5),
    ("random_mid_gamma_0.9_47x47", 0x400DD0F926949883),
    ("random_mid_blur_47x47", 0x4020793F6E13CB6C),
    ("random_mid_quantize_32_47x47", 0x3FD59E35884B6B33),
    ("random_mid_contrast_1.2_64x64", 0x4007718323A60BB0),
    ("random_mid_gamma_0.9_64x64", 0x400DDA5DD2C08F29),
    ("random_mid_blur_64x64", 0x4020CB1E508F61DD),
    ("random_mid_quantize_32_64x64", 0x3FD73E232E1E0F34),
    ("color_grad_channel_swap_16x16", 0x403759E1456B6395),
    ("color_grad_hue_shift_16x16", 0x40407036B1CF14D5),
    ("color_grad_channel_swap_23x23", 0x4040C53457F24C1D),
    ("color_grad_hue_shift_23x23", 0x4046DA7A13CD85A5),
    ("color_grad_channel_swap_32x32", 0x4046184C0D3405A1),
    ("color_grad_hue_shift_32x32", 0x404DC4081B63BE2B),
    ("color_grad_channel_swap_47x33", 0x404C9C715839F370),
    ("color_grad_hue_shift_47x33", 0x4051E7CE5E539F9B),
    ("random_color_channel_swap_32x32", 0x401182CED4BB6C67),
    ("random_color_hue_shift_32x32", 0x401CD368465F39E3),
    ("random_color_channel_swap_47x47", 0x400F504A5B38BC1D),
    ("random_color_hue_shift_47x47", 0x401D798B94E57884),
];

enum CaseResult {
    BitExact,
    WithinTolerance(f64),
    Failed(String),
}

/// Run a single test case: generate image pair, compute score, compare to x86 reference.
fn run_case(name: &str, expected_bits: u64) -> CaseResult {
    let (width, height) = match parse_dimensions(name) {
        Some(d) => d,
        None => return CaseResult::Failed(format!("{name}: could not parse dimensions")),
    };

    if width < MIN_DIMENSION || height < MIN_DIMENSION {
        return CaseResult::Failed(format!(
            "{name}: dimensions {width}x{height} below minimum {MIN_DIMENSION}"
        ));
    }

    let (img_a, img_b) = match generate_image_pair(name, width, height) {
        Some(pair) => pair,
        None => return CaseResult::Failed(format!("{name}: unknown image pattern")),
    };

    let pixels_a = rgb_bytes_to_pixels(&img_a);
    let pixels_b = rgb_bytes_to_pixels(&img_b);
    let img_a = Img::new(pixels_a, width, height);
    let img_b = Img::new(pixels_b, width, height);

    let params = ButteraugliParams::default();
    let result = match butteraugli(img_a.as_ref(), img_b.as_ref(), &params) {
        Ok(r) => r,
        Err(e) => return CaseResult::Failed(format!("{name}: butteraugli error: {e}")),
    };

    let expected = f64::from_bits(expected_bits);
    let actual = result.score;
    let actual_bits = actual.to_bits();

    if actual_bits == expected_bits {
        return CaseResult::BitExact;
    }

    // Check relative difference
    let diff = (actual - expected).abs();
    let rel = if expected.abs() > 1e-15 {
        diff / expected.abs()
    } else {
        diff
    };

    if rel > MAX_RELATIVE_DIFF {
        CaseResult::Failed(format!(
            "{name}: score mismatch — expected {expected:.10} (0x{expected_bits:016X}), \
             got {actual:.10} (0x{actual_bits:016X}), \
             rel_diff={rel:.2e} ({:.6}%)",
            rel * 100.0
        ))
    } else {
        CaseResult::WithinTolerance(rel)
    }
}

/// One-shot helper: print `(name, pnorm_3_bits)` rows for each test case
/// so they can be pasted into `X86_REFERENCE_PNORM_3`. Run with:
///   `cargo test --test cross_arch_parity capture_pnorm_3 -- --ignored --nocapture`
#[cfg(not(feature = "iir-blur"))]
#[test]
#[ignore = "capture-only; run with --ignored to regenerate X86_REFERENCE_PNORM_3"]
fn capture_pnorm_3() {
    use butteraugli::{ButteraugliParams, Img, butteraugli};
    use common::generators::{generate_image_pair, parse_dimensions, rgb_bytes_to_pixels};

    println!("// Generated by capture_pnorm_3 — paste into X86_REFERENCE_PNORM_3:");
    for &(name, _) in X86_REFERENCE_SCORES {
        let Some((w, h)) = parse_dimensions(name) else {
            continue;
        };
        let Some((a, b)) = generate_image_pair(name, w, h) else {
            continue;
        };
        let pa = rgb_bytes_to_pixels(&a);
        let pb = rgb_bytes_to_pixels(&b);
        let ia = Img::new(pa, w, h);
        let ib = Img::new(pb, w, h);
        let r = butteraugli(ia.as_ref(), ib.as_ref(), &ButteraugliParams::default()).unwrap();
        println!("    (\"{}\", 0x{:016X}),", name, r.pnorm_3.to_bits());
    }
}

/// Verify pnorm_3 matches the locked x86_64 reference values within tolerance.
/// Catches accidental cross-arch drift in the fused score reduction's f64
/// sum lanes (Σd³, Σd⁶, Σd¹²) — a different SIMD width or FMA rounding
/// strategy on ARM/AVX-512/scalar would surface here.
#[cfg(not(feature = "iir-blur"))]
#[test]
fn test_cross_arch_pnorm_3_parity() {
    use butteraugli::{ButteraugliParams, Img, butteraugli};
    use common::generators::{generate_image_pair, parse_dimensions, rgb_bytes_to_pixels};

    if X86_REFERENCE_PNORM_3.is_empty() {
        // Until the capture step has been run on the reference platform,
        // the array is empty and the test is a no-op (still compiles).
        eprintln!(
            "X86_REFERENCE_PNORM_3 is empty — run `cargo test --test cross_arch_parity \
             capture_pnorm_3 -- --ignored --nocapture` to populate."
        );
        return;
    }

    let mut bit_exact = 0usize;
    let mut within_tolerance = 0usize;
    let mut failed = 0usize;
    let mut max_rel_diff = 0.0f64;
    let mut failures = Vec::new();

    for &(name, expected_bits) in X86_REFERENCE_PNORM_3 {
        let Some((w, h)) = parse_dimensions(name) else {
            failures.push(format!("{name}: bad dims"));
            failed += 1;
            continue;
        };
        let Some((a, b)) = generate_image_pair(name, w, h) else {
            failures.push(format!("{name}: unknown pattern"));
            failed += 1;
            continue;
        };
        let pa = rgb_bytes_to_pixels(&a);
        let pb = rgb_bytes_to_pixels(&b);
        let ia = Img::new(pa, w, h);
        let ib = Img::new(pb, w, h);
        let r = match butteraugli(ia.as_ref(), ib.as_ref(), &ButteraugliParams::default()) {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!("{name}: {e}"));
                failed += 1;
                continue;
            }
        };
        let expected = f64::from_bits(expected_bits);
        let actual = r.pnorm_3;

        if actual.to_bits() == expected_bits {
            bit_exact += 1;
            continue;
        }
        let diff = (actual - expected).abs();
        let rel = if expected.abs() > 1e-15 {
            diff / expected.abs()
        } else {
            diff
        };
        if rel > MAX_RELATIVE_DIFF {
            failed += 1;
            failures.push(format!(
                "{name}: pnorm_3 mismatch — expected {expected:.10} (0x{expected_bits:016X}), \
                 got {actual:.10} (0x{:016X}), rel_diff={rel:.2e}",
                actual.to_bits()
            ));
        } else {
            within_tolerance += 1;
            if rel > max_rel_diff {
                max_rel_diff = rel;
            }
        }
    }

    let total = X86_REFERENCE_PNORM_3.len();
    eprintln!("\nCross-arch pnorm_3 parity:");
    eprintln!("  Total:            {total}");
    eprintln!("  Bit-exact:        {bit_exact}");
    eprintln!("  Within tolerance: {within_tolerance}");
    eprintln!("  Failed:           {failed}");
    if within_tolerance > 0 {
        eprintln!("  Max relative diff: {max_rel_diff:.2e}");
    }
    for msg in &failures {
        eprintln!("  FAIL: {msg}");
    }
    assert_eq!(failed, 0, "{failed}/{total} pnorm_3 parity tests failed");
}

// Reference scores were recorded against the FIR blur path. Enabling iir-blur
// changes scores by 0.1–5% on real photos and far more on tiny synthetics,
// well past this 1e-4 tolerance — gating to FIR-only.
#[cfg(not(feature = "iir-blur"))]
#[test]
fn test_cross_arch_parity_all() {
    let mut bit_exact = 0usize;
    let mut within_tolerance = 0usize;
    let mut failed = 0usize;
    let mut max_rel_diff = 0.0f64;
    let mut failures = Vec::new();

    for &(name, expected_bits) in X86_REFERENCE_SCORES {
        match run_case(name, expected_bits) {
            CaseResult::BitExact => bit_exact += 1,
            CaseResult::WithinTolerance(rel) => {
                within_tolerance += 1;
                if rel > max_rel_diff {
                    max_rel_diff = rel;
                }
            }
            CaseResult::Failed(msg) => {
                failed += 1;
                failures.push(msg);
            }
        }
    }

    let total = X86_REFERENCE_SCORES.len();
    eprintln!("\nCross-architecture parity results:");
    eprintln!("  Total:            {total}");
    eprintln!("  Bit-exact:        {bit_exact}");
    eprintln!("  Within tolerance: {within_tolerance}");
    eprintln!("  Failed:           {failed}");
    if within_tolerance > 0 {
        eprintln!("  Max relative diff: {max_rel_diff:.2e}");
    }

    for msg in &failures {
        eprintln!("  FAIL: {msg}");
    }

    assert_eq!(
        failed, 0,
        "{failed}/{total} cross-arch parity tests failed (tolerance: {MAX_RELATIVE_DIFF:.0e})"
    );
}
