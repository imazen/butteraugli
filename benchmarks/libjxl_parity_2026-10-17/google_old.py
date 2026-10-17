"""Fetch pinned Google-old sources and build the independent max-norm runners."""
import pathlib,sys,urllib.request,subprocess,shutil,hashlib
out=pathlib.Path(sys.argv[1]);(out/'butteraugli').mkdir(parents=True,exist_ok=True)
base=pathlib.Path(__file__).resolve().parent
urls={
'butteraugli.cc':'https://raw.githubusercontent.com/google/butteraugli/71b18b636b9c7d1ae0c1d3730b85b3c127eb4511/butteraugli/butteraugli.cc',
'butteraugli/butteraugli.h':'https://raw.githubusercontent.com/google/butteraugli/71b18b636b9c7d1ae0c1d3730b85b3c127eb4511/butteraugli/butteraugli.h',
'butteraugli_original.rs':'https://raw.githubusercontent.com/imazen/zenmetrics/65f536433214b104536b469eeb33a1f16781529b/crates/zenmetrics-cli/src/metrics/butteraugli_original.rs'}
for path,url in urls.items():
    p=out/path
    if not p.exists():p.write_bytes(urllib.request.urlopen(url).read())
assert hashlib.sha256((out/'butteraugli_original.rs').read_bytes()).hexdigest()=='0f4c67656458b3573be6fc5396978f7641480c7d25a3a1f4718f8ce3c597ce8f'
for f in ['cpp_main.cc','rust_main.rs']:shutil.copyfile(base/f,out/f)
subprocess.run(['rustc','--edition=2024','-O','rust_main.rs','-o','rust_old'],cwd=out,check=True)
subprocess.run(['g++','-O2','-ffp-contract=off','-I.','cpp_main.cc','butteraugli.cc','-o','cpp_old'],cwd=out,check=True)
