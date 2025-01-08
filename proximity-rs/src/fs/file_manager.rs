use std::{fs, path::Path};

pub fn read_from_file_f32(path: &Path) -> Vec<f32> {
    //todo plenty of unnecessary copying going on here
    let file_u8 = fs::read(path).unwrap();
    let chunks = file_u8.array_chunks::<516>();
    chunks.flat_map(handle_f32_raw_vec).collect::<Vec<_>>()
}

fn handle_f32_raw_vec(v: &[u8; 516]) -> Vec<f32> {
    let chunks = v[4..].array_chunks::<4>();
    chunks.map(|&chk| f32::from_le_bytes(chk)).collect()
}

pub fn read_from_npy(path: &Path) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();

    let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
    npy.into_vec().unwrap()
}
