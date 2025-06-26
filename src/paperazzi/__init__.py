# SPDX-FileCopyrightText: 2025-present MILA
#
# SPDX-License-Identifier: MIT
from paperazzi.config import CFG, Config

Config.get_global_config()

CFG.dir.log.mkdir(exist_ok=True)
