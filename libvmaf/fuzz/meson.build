if not get_option('enable_fuzz')
    subdir_done()
endif

fuzz_inc = include_directories('../tools/')

# Determine the fuzzing engine.
# When building with clang and -fsanitize=fuzzer, libFuzzer is built-in.
fuzzer_engine_flags = []
if cc.get_id() == 'clang'
    fuzzer_engine_flags = ['-fsanitize=fuzzer']
endif

fuzz_common_flags = ['-DFUZZ_BUILD']

# F1: Y4M parser fuzz target
fuzz_y4m = executable('fuzz_y4m_parser',
    ['fuzz_y4m_parser.c', '../tools/y4m_input.c'],
    include_directories: [libvmaf_inc, fuzz_inc],
    c_args: fuzz_common_flags + fuzzer_engine_flags,
    link_args: fuzzer_engine_flags,
    link_with: get_option('default_library') == 'both'
               ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [math_lib, thread_lib],
)

# F2: JSON model fuzz target
fuzz_json_model = executable('fuzz_json_model',
    ['fuzz_json_model.c'],
    include_directories: [libvmaf_inc, include_directories('../src/')],
    c_args: fuzz_common_flags + fuzzer_engine_flags,
    link_args: fuzzer_engine_flags,
    link_with: get_option('default_library') == 'both'
               ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [math_lib, thread_lib],
)

# F3: Picture read pipeline fuzz target
fuzz_read_pictures = executable('fuzz_read_pictures',
    ['fuzz_read_pictures.c'],
    include_directories: [libvmaf_inc],
    c_args: fuzz_common_flags + fuzzer_engine_flags,
    link_args: fuzzer_engine_flags,
    link_with: get_option('default_library') == 'both'
               ? libvmaf.get_static_lib() : libvmaf,
    dependencies: [math_lib, thread_lib],
)
