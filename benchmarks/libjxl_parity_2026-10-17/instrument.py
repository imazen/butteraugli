"""Apply local-only tracing to archived Rust main and the v0.12 C++ oracle."""
from pathlib import Path
home=Path.home(); rust=home/'tmp/bu-libjxl-instrumented/butteraugli/src'; cpp=home/'tmp/libjxl-v012/lib/jxl/butteraugli/butteraugli.cc'
if (rust/'parity_trace.rs').exists() or 'static void ParityOne' in cpp.read_text():
    raise SystemExit('Tracing requires fresh archived Rust main and clean libjxl v0.12 sources')
main=home/'tmp/libjxl-v012/tools/butteraugli_main.cc'
main.write_text(main.read_text().replace('printf("%g-norm: %f\\n", p, pnorm);','printf("%g-norm: %.17g\\n", p, pnorm);'))
helper=r'''
use crate::image::{ImageF, Image3F};
use std::{collections::BTreeMap, sync::{Mutex, LazyLock}, io::Write};
static COUNTS: LazyLock<Mutex<BTreeMap<String, usize>>> = LazyLock::new(||Mutex::new(BTreeMap::new()));
pub(crate) fn one(label: &str, a: &ImageF) -> Option<Vec<f32>> {
    let root=std::env::var("PARITY_DUMP").ok()?;
    let key=format!("{}_{}x{}",label,a.width(),a.height());
    let mut counts=COUNTS.lock().unwrap();let n=counts.entry(key.clone()).or_default();
    let filename=format!("{key}_{}.f32",*n);*n+=1;drop(counts);
    let mut f=std::io::BufWriter::new(std::fs::File::create(format!("{root}/{filename}")).unwrap());
    for y in 0..a.height(){for v in a.row(y){f.write_all(&v.to_le_bytes()).unwrap();}}
    if let Ok(dir)=std::env::var("PARITY_REPLACE") {
        if let Ok(bytes)=std::fs::read(format!("{dir}/{filename}")) {return Some(bytes.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect());}
    }
    None
}
pub(crate) fn three(label:&str,a:&Image3F){for c in 0..3{let _=one(&format!("{label}{c}"),a.plane(c));}}
pub(crate) fn swap(label:&str,a:&mut Image3F){for c in 0..3{if let Some(v)=one(&format!("{label}{c}"),a.plane(c)){let w=a.width();for y in 0..a.height(){a.plane_mut(c).row_mut(y).copy_from_slice(&v[y*w..(y+1)*w]);}}}}
'''
(rust/'parity_trace.rs').write_text(helper)
p=rust/'lib.rs';s=p.read_text();s+='\nmod parity_trace;\n';p.write_text(s)
p=rust/'opsin.rs';s=p.read_text();s=s.replace('    let width = rgb.plane(0).width();','    crate::parity_trace::three("input", rgb);\n    let width = rgb.plane(0).width();',1);s=s.replace('    Ok(xyb)\n}', '    crate::parity_trace::swap("opsin", &mut xyb);\n    Ok(xyb)\n}',1);p.write_text(s)
p=rust/'psycho.rs';s=p.read_text();s=s.replace('    Ok(ps)\n}', '''    crate::parity_trace::swap("lf", &mut ps.lf);
    crate::parity_trace::swap("mf", &mut ps.mf);
    for c in 0..2 { let _=crate::parity_trace::one(&format!("hf{c}"), &ps.hf[c]); let _=crate::parity_trace::one(&format!("uhf{c}"), &ps.uhf[c]); }
    Ok(ps)
}''',1);p.write_text(s)
p=rust/'diff.rs';s=p.read_text();s=s.replace('    // Add L2DiffAsymmetric for HF channels', '    let _=crate::parity_trace::one("malta0", &plane_x);\n    let _=crate::parity_trace::one("malta1", &plane_y);\n    // Add L2DiffAsymmetric for HF channels',1)
s=s.replace('    let mask = mask_psycho_image(&ps1, &ps2, Some(block_diff_ac.plane_mut(1)), &pool, &stop)?;', '''    crate::parity_trace::swap("ac_pre", &mut block_diff_ac);
    let mask = mask_psycho_image(&ps1, &ps2, Some(block_diff_ac.plane_mut(1)), &pool, &stop)?;
    let _=crate::parity_trace::one("mask", &mask);
    crate::parity_trace::swap("ac", &mut block_diff_ac);''',1)
s=s.replace('    combine_channels_to_diffmap_fused(&mask, &ps1.lf, &ps2.lf, &block_diff_ac, params.xmul(), stop)', '''    let result=combine_channels_to_diffmap_fused(&mask, &ps1.lf, &ps2.lf, &block_diff_ac, params.xmul(), stop)?;
    let _=crate::parity_trace::one("single", &result);
    Ok(result)''',1);p.write_text(s)
s=cpp.read_text().replace('#include <memory>','#include <memory>\n#include <map>\n#include <string>')
helper=r'''
static void ParityOne(const std::string& label, const ImageF& a) {
 const char* root=getenv("PARITY_DUMP");if(!root)return;
 static std::map<std::string,int> counts;
 std::string key=label+"_"+std::to_string(a.xsize())+"x"+std::to_string(a.ysize());
 std::string path=std::string(root)+"/"+key+"_"+std::to_string(counts[key]++)+".f32";
 FILE* f=fopen(path.c_str(),"wb");if(!f)abort();
 for(size_t y=0;y<a.ysize();y++){fwrite(a.ConstRow(y),4,a.xsize(),f);}
 fclose(f);
}
static void ParityThree(const std::string& label,const Image3F& a){for(int c=0;c<3;c++)ParityOne(label+std::to_string(c),a.Plane(c));}
'''
s=s.replace('namespace jxl {','namespace jxl {\n'+helper,1)
s=s.replace('  JXL_ENSURE(blurred != nullptr);','  ParityThree("input", rgb);\n  JXL_ENSURE(blurred != nullptr);',1)
start=s.index('Status OpsinDynamicsImage(');end=s.index('Status ButteraugliDiffmapInPlace',start);part=s[start:end];part=part.replace('  return true;','  ParityThree("opsin", *xyb);\n  return true;',1);s=s[:start]+part+s[end:]
start=s.index('Status SeparateFrequencies(');end=s.index('namespace {',start);part=s[start:end].replace('  return true;', '''  ParityThree("lf",ps.lf);ParityThree("mf",ps.mf);
  for(int c=0;c<2;c++){ParityOne("hf"+std::to_string(c),ps.hf[c]);ParityOne("uhf"+std::to_string(c),ps.uhf[c]);}
  return true;''',1);s=s[:start]+part+s[end:]
start=s.index('Status ButteraugliComparator::DiffmapPsychoImage(');end=s.index('double ButteraugliScoreFromDiffmap',start);part=s[start:end]
part=part.replace('  JXL_ASSIGN_OR_RETURN(Image3F block_diff_dc,','  ParityOne("malta0",block_diff_ac.Plane(0));ParityOne("malta1",block_diff_ac.Plane(1));\n  JXL_ASSIGN_OR_RETURN(Image3F block_diff_dc,',1)
part=part.replace('  ImageF mask;','  ParityThree("ac_pre",block_diff_ac);\n  ImageF mask;',1)
part=part.replace('  JXL_RETURN_IF_ERROR(HWY_DYNAMIC_DISPATCH(CombineChannelsToDiffmap)', '  ParityOne("mask",mask);ParityThree("ac",block_diff_ac);\n  JXL_RETURN_IF_ERROR(HWY_DYNAMIC_DISPATCH(CombineChannelsToDiffmap)',1)
part=part.replace('  return true;\n}', '  ParityOne("single",diffmap);\n  return true;\n}',1)
s=s[:start]+part+s[end:];cpp.write_text(s)
