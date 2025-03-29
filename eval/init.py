import argparse
import os
from pathlib import Path
from typing import get_type_hints, Any

import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from typing import Any, get_type_hints
import yaml


class BaseArgs:
    config_path = None  # Default value, can be overridden by subclasses or instances

    def __init__(self, config_path: str = None):
        self.parser = argparse.ArgumentParser(description="Model Configuration")
        self.args = None

        # Allow config_path to be set during instance creation
        if config_path is not None:
            self.config_path = config_path

        self._setup_parser()
        self._load_config()

    def _setup_parser(self) -> None:
        """通过类型注解自动设置参数解析器"""
        type_hints = get_type_hints(self.__class__)
        for name, type_ in type_hints.items():
            # Skip config_path if it's None to avoid adding None as default
            if name == 'config_path' and getattr(self.__class__, name) is None:
                self.parser.add_argument(
                    f"--{name}",
                    type=type_,
                    default=None,
                    help=f"Path to config file (类型: {type_.__name__})"
                )
                continue

            default = getattr(self.__class__, name)
            self.parser.add_argument(
                f"--{name}",
                type=type_,
                default=default,
                help=f"(默认: {default}, 类型: {type_.__name__})"
            )

    def _load_config(self) -> None:
        """从YAML文件加载配置"""
        # Use instance config_path if available, otherwise class config_path
        current_config_path = getattr(self, 'config_path', None) or self.__class__.config_path

        if not current_config_path or not Path(current_config_path).exists():
            return

        with open(current_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        type_hints = get_type_hints(self.__class__)
        for key, value in config.items():
            if not hasattr(self.__class__, key):
                inferred_type = type(value)
                setattr(self.__class__, key, None)
                self.__annotations__[key] = inferred_type
                self.parser.add_argument(
                    f"--{key}",
                    type=inferred_type,
                    default=value,
                    help=f"类型: {inferred_type.__name__}"
                )
                continue

            expected_type = type_hints.get(key, type(value))
            try:
                typed_value = expected_type(value)
                setattr(self, key, typed_value)
                for action in self.parser._actions:
                    if action.dest == key:
                        action.default = typed_value
                        break
            except (TypeError, ValueError) as e:
                print(f"警告: 配置值 '{key}' 类型不匹配: {e}")

    def parse_args(self) -> argparse.Namespace:
        """解析命令行参数"""
        self.args = self.parser.parse_args()

        # Update config_path from command line if provided
        if self.args.config_path is not None:
            self.config_path = self.args.config_path

        # Reload config if config_path was updated
        if hasattr(self.args, 'config_path') and self.args.config_path is not None:
            self._load_config()

        # 将解析后的值同步到实例属性
        for name in get_type_hints(self.__class__):
            setattr(self, name, getattr(self.args, name))
        return self.args

    def __getattr__(self, name: str) -> Any:
        """实现动态属性访问"""
        if name in get_type_hints(self.__class__):
            if self.args is None:
                self.parse_args()
            return getattr(self.args, name)
        raise AttributeError(f"'{self.__class__.__name__}' 没有属性 '{name}'")


def Init(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    if hasattr(args, 'lora_path'):
        model = PeftModel.from_pretrained(model, args.lora_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    return model, tokenizer
