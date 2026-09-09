# from __future__ import annotations

# CLASS_LIST = [
#     "yes",
#     "no",
#     "up",
#     "down",
#     "left",
#     "right",
#     "on",
#     "off",
#     "stop",
#     "go",
#     "unknown",
#     "silence",
# ]

# CLASS_ENCODING = {name: idx for idx, name in enumerate(CLASS_LIST)}

# # DSCNN 默认配置（与当前仓库中 dscnn 训练实践保持一致）
# DEFAULT_MODEL_SIZE_INFO = [
#     5,
#     64,
#     10,
#     4,
#     2,
#     2,
#     64,
#     3,
#     3,
#     1,
#     1,
#     64,
#     3,
#     3,
#     1,
#     1,
#     64,
#     3,
#     3,
#     1,
#     1,
#     64,
#     3,
#     3,
#     1,
#     1,
# ]

from __future__ import annotations

CLASS_LIST = [
    "positive",
    "negative",
]

CLASS_ENCODING = {name: idx for idx, name in enumerate(CLASS_LIST)}

DEFAULT_MODEL_SIZE_INFO = [
    5,
    64, 10, 4, 2, 2,
    64, 3, 3, 1, 1,
    64, 3, 3, 1, 1,
    64, 3, 3, 1, 1,
    64, 3, 3, 1, 1,
]