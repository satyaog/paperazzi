# SPDX-FileCopyrightText: 2025-present MILA
#
# SPDX-License-Identifier: MIT
from paperazzi.config import CFG, Config

Config.get_global_config()

LOG_DIR = CFG.dir.log
LOG_DIR.mkdir(exist_ok=True)
