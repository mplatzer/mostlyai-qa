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

import datetime
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from mostlyai.qa.accuracy import trim_label, filter_uni_acc_for_plotting, filter_biv_acc_for_plotting
from mostlyai.qa.filesystem import TemporaryWorkspace
from mostlyai.qa.assets import (
    HTML_ASSETS_PATH,
    read_html_assets,
    HTML_REPORT_TEMPLATE,
    HTML_REPORT_EARLY_EXIT,
)
from mostlyai.qa.metrics import Metrics

_LOG = logging.getLogger(__name__)


def get_uni_htmls(acc_uni: pd.DataFrame, workspace: TemporaryWorkspace) -> list[str]:
    paths_uni = workspace.get_figure_paths("univariate", acc_uni[["column"]]).values()
    return [f.read_text() for f in paths_uni]


def get_biv_htmls(acc_biv: pd.DataFrame, workspace: TemporaryWorkspace) -> tuple[list[str], list[str], list[str]]:
    acc_biv_ctx = acc_biv.loc[acc_biv.type == "ctx"]
    acc_biv_tgt = acc_biv.loc[acc_biv.type == "tgt"]
    acc_biv_nxt = acc_biv.loc[acc_biv.type == "nxt"]
    paths_biv_ctx = workspace.get_figure_paths("bivariate", acc_biv_ctx[["col1", "col2"]]).values()
    paths_biv_tgt = workspace.get_figure_paths("bivariate", acc_biv_tgt[["col1", "col2"]]).values()
    paths_biv_nxt = workspace.get_figure_paths("bivariate", acc_biv_nxt[["col1", "col2"]]).values()
    html_biv_ctx = [f.read_text() for f in paths_biv_ctx]
    html_biv_tgt = [f.read_text() for f in paths_biv_tgt]
    html_biv_nxt = [f.read_text() for f in paths_biv_nxt]
    return html_biv_ctx, html_biv_tgt, html_biv_nxt


def store_report(
    report_path: Path,
    report_type: Literal["model_report", "data_report"],
    workspace: TemporaryWorkspace,
    metrics: Metrics | None,
    meta: dict,
    acc_uni: pd.DataFrame,
    acc_biv: pd.DataFrame,
    corr_trn: pd.DataFrame,
):
    """
    Render HTML report.
    """

    # summarize accuracies by column for overview table
    accuracy_table_by_column = summarize_accuracies_by_column(acc_uni, acc_biv)
    accuracy_table_by_column = accuracy_table_by_column.sort_values("univariate", ascending=False)

    acc_uni = filter_uni_acc_for_plotting(acc_uni)
    html_uni = get_uni_htmls(acc_uni=acc_uni, workspace=workspace)
    acc_biv = filter_biv_acc_for_plotting(acc_biv, corr_trn)
    html_biv_ctx, html_biv_tgt, html_biv_nxt = get_biv_htmls(acc_biv=acc_biv, workspace=workspace)

    correlation_matrix_html_chart = workspace.get_unique_figure_path("correlation_matrices").read_text()
    similarity_pca_html_chart_path = workspace.get_unique_figure_path("similarity_pca")
    similarity_pca_html_chart = None
    if similarity_pca_html_chart_path.exists():
        similarity_pca_html_chart = similarity_pca_html_chart_path.read_text()
    if report_type == "model_report":
        accuracy_matrix_html_chart = workspace.get_unique_figure_path("accuracy_matrix").read_text()
        distances_dcr_html_chart = workspace.get_unique_figure_path("distances_dcr").read_text()
    else:
        accuracy_matrix_html_chart = None
        distances_dcr_html_chart = None

    meta |= {
        "report_creation_datetime": datetime.datetime.now(),
    }

    template = Environment(loader=FileSystemLoader(HTML_ASSETS_PATH)).get_template(HTML_REPORT_TEMPLATE)
    html = template.render(
        is_model_report=(report_type == "model_report"),
        html_assets=read_html_assets(),
        report_creation_datetime=datetime.datetime.now(),
        metrics=metrics.model_dump() if metrics else None,
        meta=meta,
        accuracy_table_by_column=accuracy_table_by_column,
        accuracy_matrix_html_chart=accuracy_matrix_html_chart,
        correlation_matrix_html_chart=correlation_matrix_html_chart,
        similarity_pca_html_chart=similarity_pca_html_chart,
        distances_dcr_html_chart=distances_dcr_html_chart,
        univariate_html_charts=html_uni,
        bivariate_html_charts_tgt=html_biv_tgt,
        bivariate_html_charts_ctx=html_biv_ctx,
        bivariate_html_charts_nxt=html_biv_nxt,
    )
    report_path.write_text(html)


def summarize_accuracies_by_column(acc_uni: pd.DataFrame, acc_biv: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates DataFrame that stores per-column univariate, bivariate and coherence accuracies.
    """

    tbl_acc_uni = acc_uni.rename(columns={"accuracy": "univariate", "accuracy_max": "univariate_max"})
    tbl_acc_biv = (
        acc_biv.loc[acc_biv.type != "nxt"]
        .groupby("col1")
        .mean(["accuracy", "accuracy_max"])
        .reset_index()
        .rename(
            columns={
                "col1": "column",
                "accuracy": "bivariate",
                "accuracy_max": "bivariate_max",
            }
        )
    )
    tbl_acc = tbl_acc_uni.merge(tbl_acc_biv, how="left")

    acc_nxt = acc_biv.loc[acc_biv.type == "nxt"]
    if not acc_nxt.empty:
        tbl_acc_coherence = (
            acc_nxt.groupby("col1")
            .mean(["accuracy", "accuracy_max"])
            .reset_index()
            .rename(
                columns={
                    "col1": "column",
                    "accuracy": "coherence",
                    "accuracy_max": "coherence_max",
                }
            )
        )
        tbl_acc = tbl_acc.merge(tbl_acc_coherence, how="left")

    tbl_acc["column"] = tbl_acc["column"].apply(lambda y: trim_label(y))
    return tbl_acc


def store_early_exit_report(report_path: Path):
    template = Environment(loader=FileSystemLoader(HTML_ASSETS_PATH)).get_template(HTML_REPORT_EARLY_EXIT)
    report_html = template.render(html_assets=read_html_assets(), meta={})
    report_path.write_text(report_html)