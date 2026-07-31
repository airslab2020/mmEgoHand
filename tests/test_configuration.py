from pathlib import Path

from mmegohand.configuration import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_paper_configuration() -> None:
    config = load_config(ROOT / "configs/paper.yaml")
    assert config.model.hidden_dim == 512
    assert config.model.active_ffn_dim == 1024
    assert config.model.legacy_ffn_dim == 2048
    assert config.model.context_decoder_layers == 30
    assert config.training.learning_rate == 1.0e-4
    assert config.training.batch_size == 32
    assert config.training.epochs == 200
    assert config.gesture.architecture == "resnet50"
    assert config.gesture.epochs == 500
