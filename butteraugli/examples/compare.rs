use butteraugli::{butteraugli, ButteraugliParams, Img, RGB8};
use std::env;
use std::fs::File;
use std::io::{BufReader, Write};

fn load_png(path: &str) -> (Vec<RGB8>, usize, usize) {
    let file = File::open(path).unwrap_or_else(|e| panic!("Failed to open {path}: {e}"));
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let bytes = &buf[..info.buffer_size()];

    let (w, h) = (info.width as usize, info.height as usize);
    let pixels = match info.color_type {
        png::ColorType::Rgb => bytes
            .chunks_exact(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect(),
        png::ColorType::Rgba => bytes
            .chunks_exact(4)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect(),
        png::ColorType::Grayscale => bytes.iter().map(|&v| RGB8::new(v, v, v)).collect(),
        _ => panic!("Unsupported color type: {:?}", info.color_type),
    };

    (pixels, w, h)
}

fn load_jpeg(path: &str) -> (Vec<RGB8>, usize, usize) {
    let file = File::open(path).unwrap_or_else(|e| panic!("Failed to open {path}: {e}"));
    let mut decoder = jpeg_decoder::Decoder::new(BufReader::new(file));
    let bytes = decoder.decode().unwrap();
    let info = decoder.info().unwrap();
    let (w, h) = (info.width as usize, info.height as usize);

    let pixels = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => bytes
            .chunks_exact(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect(),
        jpeg_decoder::PixelFormat::L8 => bytes.iter().map(|&v| RGB8::new(v, v, v)).collect(),
        _ => panic!("Unsupported JPEG format"),
    };

    (pixels, w, h)
}

fn load_image(path: &str) -> (Vec<RGB8>, usize, usize) {
    if path.ends_with(".png") {
        load_png(path)
    } else if path.ends_with(".jpg") || path.ends_with(".jpeg") {
        load_jpeg(path)
    } else {
        panic!("Unsupported format: {path}");
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: compare <image1> <image2>");
        std::process::exit(1);
    }

    let (pixels1, w1, h1) = load_image(&args[1]);
    let (pixels2, w2, h2) = load_image(&args[2]);

    assert_eq!((w1, h1), (w2, h2), "Image dimensions must match");

    let img1 = Img::new(pixels1, w1, h1);
    let img2 = Img::new(pixels2, w2, h2);
    let single_res = args.iter().any(|a| a == "--single-res");
    let params = ButteraugliParams::default()
        .with_compute_diffmap(true)
        .with_single_resolution(single_res);

    let result = butteraugli(img1.as_ref(), img2.as_ref(), &params).unwrap();
    println!("{:.6}", result.score);

    // If --rawdistmap <path> is given, dump diffmap as PFM
    if let Some(idx) = args.iter().position(|a| a == "--rawdistmap") {
        if let Some(path) = args.get(idx + 1) {
            if let Some(ref diffmap) = result.diffmap {
                let w = diffmap.width();
                let h = diffmap.height();
                let mut f = File::create(path).unwrap();
                write!(f, "Pf\n{w} {h}\n-1.0\n").unwrap();
                // PFM stores bottom-to-top
                for y in (0..h).rev() {
                    let row = &diffmap.buf()[(y * diffmap.stride())..(y * diffmap.stride() + w)];
                    for &val in row {
                        f.write_all(&val.to_le_bytes()).unwrap();
                    }
                }
                eprintln!("Wrote diffmap to {path}");
            }
        }
    }
}
