import pytest

from utils.lora_utils import parse_lora_target_modules


def test_parse_lora_target_modules_basic_list():
    assert parse_lora_target_modules("q_proj,k_proj") == ["q_proj", "k_proj"]


def test_parse_lora_target_modules_regex_pipe():
    assert parse_lora_target_modules("q_proj|k_proj|v_proj") == "q_proj|k_proj|v_proj"


def test_parse_lora_target_modules_json_list():
    assert parse_lora_target_modules('["q_proj","k_proj"]') == ["q_proj", "k_proj"]


def test_parse_lora_target_modules_regex_prefix():
    assert parse_lora_target_modules("re:q_proj|k_proj") == "q_proj|k_proj"


def test_parse_lora_target_modules_list_prefix():
    assert parse_lora_target_modules("list:q_proj|k_proj") == ["q_proj|k_proj"]


def test_lora_smoke_tiny_model():
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-MistralForCausalLM"
    )
    config = peft.LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    lora_model = peft.get_peft_model(model, config)
    trainable = [name for name, param in lora_model.named_parameters() if param.requires_grad]
    assert any("lora_" in name for name in trainable)
