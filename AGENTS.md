# Agent instructions

## Always run commands through Pixi

Run every build, check, test, doc, and example command through `pixi run`, never
with bare `cargo`. Pixi activates the toolchain defined in `pixi.toml`: the
conda-forge C/C++ compilers, CMake, Ninja (`CMAKE_GENERATOR=Ninja`), and the
`CFLAGS`/`CXXFLAGS` the CTranslate2 build expects. Building with the system
toolchain instead produces a `libctranslate2.a` that fails to link with
undefined `ctranslate2::cpu::*<CpuIsa>` symbols.

Use a task from `pixi.toml` when one fits:

```bash
pixi run check
pixi run build
pixi run test
pixi run doc
```

For anything without a task, prefix the `cargo` command with `pixi run`:

```bash
pixi run cargo test -p ct2rs -F whisper --lib
```

Pick the environment that provides the dependencies your features need:

- `hub` feature (model downloads, Whisper tests that fetch models): add
  `-e hub`, for example
  `pixi run -e hub cargo test -p ct2rs -F whisper,hub`.
- CUDA builds: use the `cuda-dyn`, `cuda-full`, `whisper-dyn`, or
  `check-whisper-dyn` tasks, which select their own environment and
  `CARGO_TARGET_DIR`.
- `ct2rs-platform`: use the `platform` environment (`-e platform`).
