import json
from typing import Dict, Any
from abc import ABC, abstractmethod


class DataParser(ABC):
    """抽象解析器"""

    @abstractmethod
    def parse(self, data: str) -> Dict[str, Any]:
        pass


class JsonDataParser(DataParser):
    """JSON数据解析器"""

    def parse(self, data: str) -> Dict[str, Any]:
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON parsing failed: {e}") from e
