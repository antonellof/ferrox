# frink-gguf

GGUF mmap loader for Frink.

Reads the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
binary format (metadata + tensor blobs) via memory-mapped files so quantized
weights stay on disk until kernels touch them.
