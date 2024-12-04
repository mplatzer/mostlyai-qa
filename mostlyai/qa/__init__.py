# Copyright 2024 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pandas as pd
from packaging.version import Version

from mostlyai.qa.report import report
from mostlyai.qa.report_from_statistics import report_from_statistics

__all__ = ["report", "report_from_statistics"]
__version__ = "1.3.0"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if Version(pd.__version__) >= Version("2.2.0"):
    pd.set_option("future.no_silent_downcasting", True)