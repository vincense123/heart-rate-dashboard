"""
Assemble the dashboard.

  python3 web/build.py

Reads   web/dashboard.src.html + web/ppg_core.js
Writes  index.html          standalone page (GitHub Pages / any web server)
        web/dist/artifact.html  page body only, for publishing as a hosted artifact
"""
import pathlib

root = pathlib.Path(__file__).resolve().parent
src = (root / "dashboard.src.html").read_text()
core = (root / "ppg_core.js").read_text()

fragment = src.replace("<!--CORE-->", "<script>\n" + core + "\n</script>")

wrapper = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
:root{color-scheme:light}
body{margin:0}
[hidden]{display:none!important}
</style>
</head>
<body>
%s
</body>
</html>
"""

(root.parent / "index.html").write_text(wrapper % fragment)
(root / "dist").mkdir(exist_ok=True)
(root / "dist" / "artifact.html").write_text(fragment)
print("built", len(fragment) // 1024, "KB")
