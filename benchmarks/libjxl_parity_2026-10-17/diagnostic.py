from pathlib import Path
import sys
root=Path.home()/'tmp/bu-libjxl-instrumented/butteraugli/src';p=root/'blur.rs';base=p.with_suffix('.rs.baseline')
if not base.exists():base.write_text(p.read_text())
s=base.read_text()
s=s.replace('let scaler = -1.0 / (2.0 * sigma * sigma);','let scaler = -1.0 / (2.0 * f64::from(sigma) * f64::from(sigma));',1)
s=s.replace('let weight = (scaler * (i * i) as f32).exp();','let weight = (scaler * f64::from(i) * f64::from(i)).exp() as f32;',1)
if sys.argv[1]=='strict':
    idx=s.index('pub(crate) fn gaussian_blur_stop(');brace=s.index('{',idx)
    # Dispatch the common non-5 kernels through C++'s exact summation tree.
    s=s[:brace+1]+'''
    if sigma != 1.2 {
        let kernel=compute_kernel(sigma);
        if kernel.len()!=5 {return Ok(parity_blur(input,&kernel));}
    }
'''+s[brace+1:]
    s+='''
fn parity_blur(input:&ImageF,kernel:&[f32])->ImageF {
    fn pass(input:&ImageF,kernel:&[f32])->ImageF {
        let n=kernel.len();let radius=n/2;let mut sum=0.0_f32;for &v in kernel {sum+=v;}let scale=1.0/sum;
        let sk:Vec<f32>=kernel.iter().map(|&v|v*scale).collect();
        let mut out=ImageF::new(input.height(),input.width());
        for y in 0..input.height(){let row=input.row(y);for x in 0..input.width(){
            let v=if x<radius || x+radius>=input.width(){
                let lo=x.saturating_sub(radius);let hi=(x+radius).min(input.width()-1);let mut z=0.0_f32;let mut v=0.0_f32;
                for j in lo..=hi{z+=kernel[j+radius-x];v+=row[j]*kernel[j+radius-x];}v*(1.0/z)
            }else{let row=&row[x-radius..x+radius+1];
                if n==7 {row[3]*sk[3]+(row[0]+row[6])*sk[0]+(row[1]+row[5])*sk[1]+(row[2]+row[4])*sk[2]}
                else{let mut sums=[0.0_f32;4];for i in 0..radius{let t=(row[i]+row[n-1-i])*sk[i];if i<4{sums[i]=t;}else{sums[i%4]+=t;}}row[radius]*sk[radius]+sums[0]+sums[1]+sums[2]+sums[3]}
            };out.row_mut(x)[y]=v;
        }}out
    }
    pass(&pass(input,kernel),kernel)
}
'''
p.write_text(s)
