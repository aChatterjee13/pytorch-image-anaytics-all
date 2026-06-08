from image_analytics.serving.triton_config import (
    TritonModelConfig,
    generate_triton_config,
    write_triton_config,
)


class TestTritonConfig:
    def test_renders_core_fields(self):
        text = generate_triton_config(
            TritonModelConfig(name="clf", input_dims=[3, 224, 224], output_dims=[10])
        )
        assert 'name: "clf"' in text
        assert 'backend: "onnxruntime"' in text
        assert "max_batch_size: 8" in text
        assert "dims: [ 3, 224, 224 ]" in text
        assert "dims: [ 10 ]" in text
        assert "dynamic_batching {" in text
        assert "instance_group [" in text
        assert "kind: KIND_CPU" in text

    def test_dynamic_batching_toggle(self):
        text = generate_triton_config(
            TritonModelConfig(name="x", input_dims=[3], output_dims=[2], dynamic_batching=False)
        )
        assert "dynamic_batching" not in text

    def test_custom_io_names_and_gpu(self):
        text = generate_triton_config(
            TritonModelConfig(
                name="det", input_dims=[3, 96, 96], output_dims=[100, 4],
                input_name="images", output_name="boxes",
                instance_kind="KIND_GPU", instance_count=2,
            )
        )
        assert 'name: "images"' in text and 'name: "boxes"' in text
        assert "kind: KIND_GPU" in text and "count: 2" in text

    def test_write_to_disk(self, tmp_path):
        cfg = TritonModelConfig(name="m", input_dims=[3, 64, 64], output_dims=[4])
        path = tmp_path / "models" / "m" / "config.pbtxt"
        text = write_triton_config(cfg, path)
        assert path.exists() and path.read_text() == text
