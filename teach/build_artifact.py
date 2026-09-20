#!/usr/bin/env python3
"""把 lessons/ 或 reference/ 下的课程页打包成自包含单文件,用于发布成可在线打开的页面。

为什么需要它:课程页按共享组件写(link 到 assets/lesson.css、script 到 assets/quiz.js),
本地看和打印都好;但发布出去的页面是**单个文件**,而且外部请求被 CSP 挡掉,相对路径也不存在。
所以发布前把 CSS/JS 内联进去。一份源文件,两种产物,不维护两份内容。

    python teach/build_artifact.py teach/lessons/0001-four-routes.html
    → teach/build/0001-four-routes.html

同时做两件收尾:
- 去掉 <meta charset>(发布时外层会自己套 doctype/head/body,只保留 <title>);
- 把指向本地文件的相对链接拆掉(只留文字),避免发布后出现点不开的死链。
"""

import re
import sys
from pathlib import Path


def inline(html: str, base: Path) -> str:
    def css(m):
        text = (base / m.group(1)).read_text(encoding="utf-8")
        return f"<style>\n{text}\n</style>"

    def js(m):
        text = (base / m.group(1)).read_text(encoding="utf-8")
        return f"<script>\n{text}\n</script>"

    html = re.sub(r'<link rel="stylesheet" href="([^"]+)">', css, html)
    html = re.sub(r'<script src="([^"]+)"></script>', js, html)
    html = re.sub(r'<meta charset="[^"]+">\s*', "", html)
    # 相对链接 → 纯文字(保留内容,去掉死链)
    html = re.sub(r'<a href="(?!https?:)[^"]*">(.*?)</a>', r"\1", html, flags=re.S)
    return html


def main() -> None:
    src = Path(sys.argv[1])
    out_dir = Path(__file__).parent / "build"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / src.name
    out.write_text(inline(src.read_text(encoding="utf-8"), src.parent), encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
