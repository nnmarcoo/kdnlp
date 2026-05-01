use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // OUT_DIR is target/<profile>/build/<crate>/out — go up 3 levels to get target/<profile>/
    let target_dir = out_dir.ancestors().nth(3).unwrap();

    // Copy ORT library for the current platform
    #[cfg(target_os = "windows")]
    copy("lib/windows-x64/onnxruntime.dll", &target_dir.join("onnxruntime.dll"));

    #[cfg(target_os = "linux")]
    copy("lib/linux-x64/libonnxruntime.so", &target_dir.join("libonnxruntime.so"));

    #[cfg(target_os = "macos")]
    copy("lib/macos-arm64/libonnxruntime.dylib", &target_dir.join("libonnxruntime.dylib"));

    // Copy model files next to the exe
    let model_out = target_dir.join("model");
    std::fs::create_dir_all(&model_out).unwrap();
    copy("model/embedder.onnx",    &model_out.join("embedder.onnx"));
    copy("model/norm_stats.json",  &model_out.join("norm_stats.json"));
}

fn copy(src: &str, dst: &PathBuf) {
    if std::path::Path::new(src).exists() {
        std::fs::copy(src, dst).unwrap();
    }
}
