use std::{fs, path::Path};

pub fn read_from_file_f32(path: &Path) -> Vec<f32> {
    let file_u8 = fs::read(path).unwrap();

    // Each record is 516 bytes: 4-byte header + 512 bytes (128 f32)
    let mut out = Vec::with_capacity(file_u8.len() / 516 * 128);

    for rec in file_u8.chunks_exact(516) {
        // skip the first 4 bytes
        for b in rec[4..].chunks_exact(4) {
            // safe because chunks_exact(4) yields exactly 4 bytes
            out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        }
    }
    out
}

pub fn read_from_npy(path: &Path) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();
    let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
    npy.into_vec().unwrap()
}
