"""配置目录中的机器人 I/O、策略契约与控制默认值。仅本地配置可指定驱动。"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StateField(ConfigModel):
    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    size: int = Field(gt=0, strict=True)
    unit: str = ""


class ActionField(ConfigModel):
    name: str = Field(min_length=1)
    size: int = Field(gt=0, strict=True)
    unit: str = ""


class RobotObservation(ConfigModel):
    # 逻辑相机名 -> 驱动提供的源字段；图像方向由驱动统一为正向 RGB。
    images: dict[str, str]
    state: list[StateField] = Field(min_length=1)
    instruction: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_fields(self):
        if any(not name or not source for name, source in self.images.items()):
            raise ValueError("相机名称与源字段不能为空")
        names = [part.name for part in self.state]
        if len(names) != len(set(names)):
            raise ValueError("状态字段名称不能重复")
        return self

    @property
    def state_dim(self) -> int:
        return sum(part.size for part in self.state)


class RobotAction(ConfigModel):
    space: str = Field(min_length=1)
    fields: list[ActionField] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_fields(self):
        names = [part.name for part in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("动作字段名称不能重复")
        return self

    @property
    def dim(self) -> int:
        return sum(part.size for part in self.fields)


class RobotConfig(ConfigModel):
    name: str = Field(min_length=1)
    driver: str = Field(pattern=r"^[\w.]+:[\w]+$")
    kind: Literal["simulation", "hardware"]
    observation: RobotObservation
    action: RobotAction
    options: dict[str, Any] = Field(default_factory=dict)


class PolicyInputs(ConfigModel):
    # 逻辑相机名 -> 模型请求字段，支持任意数量、任意命名的相机。
    images: dict[str, str]
    state_key: str = Field(min_length=1)
    instruction_key: str = Field(min_length=1)
    state_dim: int = Field(gt=0, strict=True)
    state_indices: list[Annotated[int, Field(ge=0, strict=True)]] | None = None

    @model_validator(mode="after")
    def validate_mapping(self):
        keys = [*self.images.values(), self.state_key, self.instruction_key]
        if any(not name for name in self.images) or any(not key for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("策略请求字段不能为空或重复")
        if self.state_indices is not None:
            if len(self.state_indices) != self.state_dim:
                raise ValueError("state_indices 长度必须等于 state_dim")
        return self


class PolicyOutputs(ConfigModel):
    action_space: str = Field(min_length=1)
    action_dim: int = Field(gt=0, strict=True)
    actions_key: str = Field(default="actions", min_length=1)


class PolicyConfig(ConfigModel):
    name: str = Field(min_length=1)
    driver: str = Field(pattern=r"^[\w.]+:[\w]+$")
    inputs: PolicyInputs
    outputs: PolicyOutputs
    options: dict[str, Any] = Field(default_factory=dict)


class ControlDefaults(ConfigModel):
    robot: str = Field(min_length=1)
    policy: str = Field(min_length=1)
    max_steps: int = Field(default=300, ge=1, le=10000, strict=True)
    replan_steps: int = Field(default=5, ge=1, le=100, strict=True)


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match):
            name, default = match.groups()
            result = os.getenv(name)
            if result is not None and (result or default is None):
                return result
            if default is not None:
                return default
            raise ValueError(f"配置需要环境变量 {name}")
        return _ENV.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def read_config(path: Path, model: type[ConfigModel]) -> Any:
    # 先解析 YAML 再替换值，避免密码等内容被解释为 YAML 语法。
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"配置必须为 YAML 对象：{path}")
    try:
        return model.model_validate(_expand_env(data))
    except ValidationError as exc:
        # 验证错误可能含凭据值，避免把原始输入送进 Agent 的工具响应。
        details = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors(include_input=False, include_context=False)
        )
        raise ValueError(f"配置格式错误：{path}：{details}") from None
