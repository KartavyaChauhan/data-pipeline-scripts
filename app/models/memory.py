from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MemoryManager:
	system_prompt: str = "You are a helpful assistant."
	max_messages: int = 20
	_messages: List[Dict[str, Any]] = field(default_factory=list)

	def add_message(self, role: str, content: str) -> None:
		self._messages.append({"role": role, "content": content})
		if self.max_messages > 0 and len(self._messages) > self.max_messages:
			self._messages = self._messages[-self.max_messages :]

	def get_context(self) -> List[Dict[str, Any]]:
		context = [{"role": "system", "content": self.system_prompt}]
		context.extend(self._messages)
		return context

	def clear(self) -> None:
		self._messages.clear()
