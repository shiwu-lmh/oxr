from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.postprocess.table_postprocess import postprocess_table
from oxr.utils.table import html_table_to_markdown


def _table_block() -> Block:
    return Block(0, BlockType.TABLE, [0, 0, 10, 10], "", ContentFormat.TEXT)


def test_html_table_to_markdown_expands_rowspan_and_colspan():
    html = """
    <table>
      <tr><th rowspan="2">Name</th><th colspan="2">Score</th></tr>
      <tr><td>Math</td><td>English</td></tr>
      <tr><td>Ada</td><td>98</td><td>96</td></tr>
    </table>
    """

    assert html_table_to_markdown(html) == "\n".join([
        "| Name | Score | Score |",
        "| --- | --- | --- |",
        "| Name | Math | English |",
        "| Ada | 98 | 96 |",
    ])


def test_html_table_to_markdown_escapes_pipes_and_strips_fences():
    html = """```html
    <table><tr><th>A|B</th></tr><tr><td>x|y</td></tr></table>
    ```"""

    assert html_table_to_markdown(html) == "\n".join([
        "| A\\|B |",
        "| --- |",
        "| x\\|y |",
    ])


def test_html_table_to_markdown_handles_thead_tbody_and_th():
    html = """
    <table>
      <thead>
        <tr><th>City</th><th>Population</th></tr>
      </thead>
      <tbody>
        <tr><td>Shanghai</td><td>24M</td></tr>
        <tr><td>Beijing</td><td>21M</td></tr>
      </tbody>
    </table>
    """

    assert html_table_to_markdown(html) == "\n".join([
        "| City | Population |",
        "| --- | --- |",
        "| Shanghai | 24M |",
        "| Beijing | 21M |",
    ])


def test_html_table_to_markdown_falls_back_to_cleaned_text():
    assert html_table_to_markdown("<p>not a table</p>") == "not a table"


def test_postprocess_table_sets_markdown_format_when_requested():
    block = postprocess_table(
        _table_block(),
        "<table><tr><th>A</th></tr><tr><td>B</td></tr></table>",
        output_format="markdown",
    )

    assert block.format == ContentFormat.MARKDOWN
    assert block.content == "| A |\n| --- |\n| B |"


def test_postprocess_table_attaches_abnormal_arrow_to_result_and_restores_columns():
    raw = (
        "<table><thead><tr><th>代号</th><th>项目名称</th><th>结果</th>"
        "<th>参考区间</th><th>单位</th><th>检测方法</th><th>互认</th></tr></thead>"
        "<tbody><tr><td>NEU%</td><td>中性粒细胞百分比</td><td>76.2</td>"
        "<td>↑</td><td>40-75</td><td>%</td><td>计算法</td></tr></tbody></table>"
    )

    block = postprocess_table(_table_block(), raw)

    assert "<td>76.2 ↑</td><td>40-75</td><td>%</td><td>计算法</td><td></td>" in block.content


def test_postprocess_table_does_not_merge_arrows_into_reference_values():
    raw = (
        "<table><tr><th>项目</th><th>结果</th><th>参考区间</th></tr>"
        "<tr><td>某指标</td><td>↑</td><td>0-1</td></tr></table>"
    )

    block = postprocess_table(_table_block(), raw)

    assert "<td>↑</td><td>0-1</td>" in block.content
