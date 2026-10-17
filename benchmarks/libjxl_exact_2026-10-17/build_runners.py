"""Build baseline and opt-in runners into separate Cargo output directories."""
import os,pathlib,shutil,subprocess
home=pathlib.Path.home();root=home/'tmp/bu-libjxl-data/round2-final';root.mkdir(parents=True,exist_ok=True)
work=pathlib.Path(__file__).resolve().parents[2]
for label,manifest,target in [('origin-main',home/'tmp/bu-libjxl-r2-baseline/Cargo.toml',root/'baseline-target'),('exact',work/'Cargo.toml',root/'current-target')]:
    subprocess.run(['cargo','build','--release','--manifest-path',str(manifest),'--target-dir',str(target),'-p','butteraugli-cli','--example','libjxl_parity'],check=True)
    shutil.copyfile(target/'release/examples/libjxl_parity',root/f'{label}-runner');(root/f'{label}-runner').chmod(0o755)
